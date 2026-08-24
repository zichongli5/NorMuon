import importlib.util
import sys
import types
import unittest
from unittest import mock

import torch

import normuon


KJ = (3.4445, -4.7750, 2.0315)
HAVE_M512_TRITON = torch.cuda.is_available() and importlib.util.find_spec("triton") is not None


class DispatchTest(unittest.TestCase):
    def test_default_path_is_unchanged(self):
        torch.manual_seed(1)
        grad = torch.randn(32, 16)
        expected = normuon.zeropower_via_newtonschulz5(grad.clone())
        actual = normuon.zeropower_via_newtonschulz5(grad.clone(), use_triton=False)
        self.assertTrue(torch.equal(actual, expected))

    def test_m512_dispatches_and_propagates_steps(self):
        calls = []

        def fake_kernel(value, steps):
            calls.append((value.shape, steps))
            return value

        module = types.SimpleNamespace(zeropower_via_newtonschulz5_triton=fake_kernel)
        value = torch.randn(512, 512)
        with mock.patch.dict(sys.modules, {"ns_triton": module}):
            actual = normuon.zeropower_via_newtonschulz5(value, steps=5, use_triton=True)
        self.assertIs(actual, value)
        self.assertEqual(calls, [(torch.Size([512, 512]), 5)])

    def test_non_m512_retains_torch_path(self):
        module = types.SimpleNamespace(
            zeropower_via_newtonschulz5_triton=mock.Mock(
                side_effect=AssertionError("Triton must not run")
            )
        )
        for shape in ((128, 64), (128, 128), (128, 512)):
            with self.subTest(shape=shape):
                value = torch.randn(*shape)
                with mock.patch.dict(sys.modules, {"ns_triton": module}):
                    actual = normuon.zeropower_via_newtonschulz5(value, use_triton=True)
                expected = normuon.zeropower_via_newtonschulz5(value)
                self.assertTrue(torch.equal(actual, expected))
        module.zeropower_via_newtonschulz5_triton.assert_not_called()

    def test_missing_optional_dependency_fails_clearly(self):
        with mock.patch.dict(sys.modules, {"ns_triton": None}):
            with self.assertRaisesRegex(RuntimeError, "optional Triton dependency"):
                normuon.zeropower_via_newtonschulz5(
                    torch.randn(512, 512), use_triton=True
                )

    def test_optimizer_defaults_off(self):
        parameter = torch.nn.Parameter(torch.randn(8, 8))
        optimizer = normuon.SingleDeviceNorMuon([parameter])
        self.assertFalse(optimizer.param_groups[0]["use_triton"])

    def test_aux_optimizer_routes_opt_in_muon_group(self):
        muon = torch.nn.Parameter(torch.randn(8, 8))
        adam = torch.nn.Parameter(torch.randn(8))
        optimizer = normuon.SingleDeviceNorMuonWithAuxAdam([
            {"params": [muon], "use_muon": True, "use_triton": True},
            {"params": [adam], "use_muon": False},
        ])
        with mock.patch.object(normuon, "_single_device_triton_step") as batched_step:
            optimizer.step()
        batched_step.assert_called_once_with(optimizer, optimizer.param_groups[0])


@unittest.skipUnless(HAVE_M512_TRITON, "M=512 Triton tests require CUDA and Triton")
class M512TritonTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import ns_triton

        cls.kernel = ns_triton

    def test_line_1(self):
        torch.manual_seed(2)
        value = torch.randn(512, 768, device="cuda", dtype=torch.bfloat16)
        actual = self.kernel.ns_line_1(value)
        expected = value @ value.mT
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        self.assertTrue(torch.equal(actual, actual.mT))

    def test_output_must_not_alias_input(self):
        value = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "must not share storage"):
            self.kernel.ns_line_1(value, out=value)

    def test_line_2(self):
        torch.manual_seed(3)
        value = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        value = (value + value.mT) / 2
        actual = self.kernel.ns_line_2(value, alpha=KJ[2], beta=KJ[1])
        expected = KJ[2] * (value @ value.mT) + KJ[1] * value
        relative_l2 = ((actual.float() - expected.float()).norm()
                       / expected.float().norm()).item()
        self.assertLess(relative_l2, 0.005)
        self.assertTrue(torch.equal(actual, actual.mT))

    def test_line_3(self):
        torch.manual_seed(4)
        left = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        right = torch.randn(512, 768, device="cuda", dtype=torch.bfloat16)
        actual = self.kernel.ns_line_3(left, right, KJ[0])
        expected = KJ[0] * right + left @ right
        relative_l2 = ((actual.float() - expected.float()).norm()
                       / expected.float().norm()).item()
        self.assertLess(relative_l2, 0.005)

    @staticmethod
    def _fp32_kj(value):
        a, b, c = KJ
        transposed = value.size(-2) > value.size(-1)
        result = value.float().mT if transposed else value.float()
        result = result / (result.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(5):
            gram = result @ result.mT
            result = a * result + (b * gram + c * (gram @ gram)) @ result
        return result.mT if transposed else result

    def test_end_to_end_parity_and_layout(self):
        for shape in ((512, 512), (2560, 512), (512, 2560)):
            with self.subTest(shape=shape):
                torch.manual_seed(sum(shape))
                value = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
                expected = normuon.zeropower_via_newtonschulz5(value)
                actual = self.kernel.zeropower_via_newtonschulz5_triton(value)
                fp32 = self._fp32_kj(value)
                torch_error = ((expected.float() - fp32).norm() / fp32.norm()).item()
                triton_error = ((actual.float() - fp32).norm() / fp32.norm()).item()
                self.assertLess(triton_error, 0.02)
                self.assertLessEqual(triton_error, 2 * torch_error)
                self.assertEqual(actual.shape, value.shape)
                self.assertTrue(actual.is_contiguous())
                self.assertTrue(torch.isfinite(actual).all())

    def test_deterministic(self):
        torch.manual_seed(5)
        value = torch.randn(2, 512, 768, device="cuda", dtype=torch.bfloat16)
        first = self.kernel.zeropower_via_newtonschulz5_triton(value)
        second = self.kernel.zeropower_via_newtonschulz5_triton(value)
        self.assertTrue(torch.equal(first, second))

    def test_optimizer_update_parity(self):
        torch.manual_seed(6)
        grad = torch.randn(512, 768, device="cuda", dtype=torch.bfloat16)
        torch_update = normuon.normuon_update(
            grad.clone(), torch.zeros_like(grad), torch.zeros_like(grad[..., 0:1])
        )
        triton_update = normuon.normuon_update(
            grad.clone(),
            torch.zeros_like(grad),
            torch.zeros_like(grad[..., 0:1]),
            use_triton=True,
        )
        relative_l2 = ((triton_update.float() - torch_update.float()).norm()
                       / torch_update.float().norm()).item()
        self.assertLess(relative_l2, 0.03)

    def test_batched_optimizer_updates_parameters_and_state(self):
        torch.manual_seed(7)
        lr, weight_decay = 0.02, 0.1
        params = [
            torch.nn.Parameter(torch.randn(512, 768, device="cuda", dtype=torch.bfloat16))
            for _ in range(2)
        ]
        grads = [torch.randn_like(p) for p in params]
        expected_updates = []
        expected_momentum = []
        expected_second = []
        for grad in grads:
            momentum = torch.zeros_like(grad)
            second = torch.zeros_like(grad[..., 0:1])
            expected_updates.append(
                normuon.normuon_update(grad.clone(), momentum, second, use_triton=True)
            )
            expected_momentum.append(momentum)
            expected_second.append(second)
        expected_params = []
        for p, update in zip(params, expected_updates):
            expected = p.detach().clone()
            expected.mul_(1 - lr * weight_decay)
            expected.add_(update, alpha=-lr)
            expected_params.append(expected)

        optimizer = normuon.SingleDeviceNorMuon(
            params, lr=lr, weight_decay=weight_decay, use_triton=True
        )
        for p, grad in zip(params, grads):
            p.grad = grad
        optimizer.step()

        for index, (p, expected) in enumerate(zip(params, expected_params)):
            self.assertTrue(torch.equal(p, expected))
            self.assertTrue(torch.equal(
                optimizer.state[p]["momentum_buffer"], expected_momentum[index]
            ))
            self.assertTrue(torch.equal(
                optimizer.state[p]["second_momentum_buffer"], expected_second[index]
            ))


if __name__ == "__main__":
    unittest.main()
