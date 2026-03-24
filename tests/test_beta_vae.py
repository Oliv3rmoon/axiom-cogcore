"""Tests for the β-VAE module."""
from __future__ import annotations

import pytest
import torch
import numpy as np

import config
from beta_vae.model import BetaVAE, BetaVAEEncoder, BetaVAEDecoder
from beta_vae.trainer import BetaVAETrainer

INPUT_DIM = config.HOPFIELD_PATTERN_DIM


class TestBetaVAEEncoder:
    def test_output_shapes(self):
        enc = BetaVAEEncoder(input_dim=INPUT_DIM).to(config.DEVICE)
        x = torch.randn(4, INPUT_DIM, device=config.DEVICE)
        mu, logvar = enc(x)
        assert mu.shape == (4, config.BETA_VAE_LATENT_DIM)
        assert logvar.shape == (4, config.BETA_VAE_LATENT_DIM)

    def test_logvar_clamped(self):
        enc = BetaVAEEncoder(input_dim=INPUT_DIM).to(config.DEVICE)
        x = torch.randn(4, INPUT_DIM, device=config.DEVICE) * 100  # Large input
        _, logvar = enc(x)
        assert logvar.max().item() <= 2.0
        assert logvar.min().item() >= -10.0


class TestBetaVAEDecoder:
    def test_output_shape(self):
        dec = BetaVAEDecoder(output_dim=INPUT_DIM).to(config.DEVICE)
        z = torch.randn(4, config.BETA_VAE_LATENT_DIM, device=config.DEVICE)
        out = dec(z)
        assert out.shape == (4, INPUT_DIM)


class TestBetaVAE:
    @pytest.fixture
    def model(self):
        return BetaVAE(input_dim=INPUT_DIM).to(config.DEVICE)

    def test_forward(self, model):
        x = torch.randn(4, INPUT_DIM, device=config.DEVICE)
        x_recon, x_orig, mu, logvar = model(x)
        assert x_recon.shape == x.shape
        assert torch.equal(x_orig, x)
        assert mu.shape == (4, config.BETA_VAE_LATENT_DIM)

    def test_encode_decode_roundtrip(self, model):
        x = torch.randn(4, INPUT_DIM, device=config.DEVICE)
        mu, logvar = model.encode(x)
        z = model.reparameterize(mu, logvar)
        x_recon = model.decode(z)
        assert x_recon.shape == x.shape

    def test_loss_function(self, model):
        x = torch.randn(8, INPUT_DIM, device=config.DEVICE)
        x_recon, x_orig, mu, logvar = model(x)
        losses = BetaVAE.loss_function(x_recon, x_orig, mu, logvar)
        assert "loss" in losses
        assert "recon_loss" in losses
        assert "kl" in losses
        assert losses["loss"].requires_grad

    def test_loss_with_capacity(self, model):
        x = torch.randn(8, INPUT_DIM, device=config.DEVICE)
        x_recon, x_orig, mu, logvar = model(x)
        losses_c0 = BetaVAE.loss_function(x_recon, x_orig, mu, logvar, capacity=0.0)
        losses_c25 = BetaVAE.loss_function(x_recon, x_orig, mu, logvar, capacity=25.0)
        # Different capacity should give different losses
        assert losses_c0["loss"].item() != losses_c25["loss"].item()

    def test_backward(self, model):
        model.train()
        x = torch.randn(4, INPUT_DIM, device=config.DEVICE)
        x_recon, x_orig, mu, logvar = model(x)
        losses = BetaVAE.loss_function(x_recon, x_orig, mu, logvar)
        losses["loss"].backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_reparameterize_deterministic_in_eval(self, model):
        model.eval()
        mu = torch.randn(4, config.BETA_VAE_LATENT_DIM, device=config.DEVICE)
        logvar = torch.zeros(4, config.BETA_VAE_LATENT_DIM, device=config.DEVICE)
        z = model.reparameterize(mu, logvar)
        assert torch.allclose(z, mu)


class TestBetaVAETrainer:
    def test_train_step(self):
        model = BetaVAE(input_dim=INPUT_DIM).to(config.DEVICE)
        trainer = BetaVAETrainer(model)
        batch = torch.randn(16, INPUT_DIM, device=config.DEVICE)
        result = trainer.train_step(batch)
        assert "loss" in result
        assert "recon_loss" in result
        assert "kl" in result
        assert result["step"] == 1

    def test_capacity_annealing(self):
        model = BetaVAE(input_dim=INPUT_DIM).to(config.DEVICE)
        trainer = BetaVAETrainer(model)
        batch = torch.randn(16, INPUT_DIM, device=config.DEVICE)
        trainer.train_step(batch)
        assert trainer._capacity > 0.0

    def test_loss_decreases(self):
        model = BetaVAE(input_dim=INPUT_DIM).to(config.DEVICE)
        trainer = BetaVAETrainer(model)
        batch = torch.randn(32, INPUT_DIM, device=config.DEVICE)
        losses = []
        for _ in range(20):
            result = trainer.train_step(batch)
            losses.append(result["loss"])
        # Loss should generally decrease (first vs last 5)
        first_avg = np.mean(losses[:5])
        last_avg = np.mean(losses[-5:])
        assert last_avg < first_avg

    def test_stats(self):
        model = BetaVAE(input_dim=INPUT_DIM).to(config.DEVICE)
        trainer = BetaVAETrainer(model)
        stats = trainer.get_stats()
        assert stats["latent_dim"] == config.BETA_VAE_LATENT_DIM
        assert stats["beta"] == config.BETA_VAE_BETA
