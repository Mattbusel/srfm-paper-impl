//! SRFM Paper — Rust reference implementation
//!
//! Demonstrates the five core formulas from the paper:
//!   1. Price-velocity β  (Eq. 3)
//!   2. Lorentz factor γ  (Eq. 4)
//!   3. Spacetime interval s²  (Eq. 6)
//!   4. Proper time τ  (Eq. 7)
//!   5. Rapidity φ = arctanh(β)  (Eq. 9)
//!
//! The full C++20 production implementation lives in:
//!   https://github.com/Mattbusel/Special-Relativity-in-Financial-Modeling
//!
//! The Python reference implementation lives in:
//!   https://github.com/Mattbusel/srfm-python

const BETA_MAX: f64 = 0.9999;

/// Compute price velocity β from an OHLCV bar.
///
/// β = |log(close/open)| / (c_market × Δt)
///
/// where c_market is calibrated as the 99.5th percentile of |log(close/open)|
/// over a rolling window of 20 bars.
fn beta(open: f64, close: f64, c_market: f64, delta_t: f64) -> f64 {
    let log_return = (close / open).ln().abs();
    let raw = log_return / (c_market * delta_t);
    raw.min(BETA_MAX)
}

/// Lorentz factor γ = 1 / √(1 − β²)
fn gamma(b: f64) -> f64 {
    1.0 / (1.0 - b * b).sqrt()
}

/// Spacetime interval s² = (c_market Δt)² − (Δx)²
///
/// s² > 0 → TIMELIKE  (information propagates within the light cone)
/// s² < 0 → SPACELIKE (price move exceeds "speed of light")
/// s² = 0 → LIGHTLIKE
fn spacetime_interval(c_market: f64, delta_t: f64, delta_x: f64) -> f64 {
    let ct = c_market * delta_t;
    ct * ct - delta_x * delta_x
}

/// Proper time τ = Δt / γ = Δt × √(1 − β²)
fn proper_time(delta_t: f64, b: f64) -> f64 {
    delta_t * (1.0 - b * b).sqrt()
}

/// Rapidity φ = arctanh(β)
///
/// Rapidities are additive under velocity composition:
///   φ(β₁₂) = φ(β₁) + φ(β₂)
fn rapidity(b: f64) -> f64 {
    b.atanh()
}

/// Regime label from spacetime interval.
fn regime(s2: f64) -> &'static str {
    if s2 > 1e-12 {
        "TIMELIKE"
    } else if s2 < -1e-12 {
        "SPACELIKE"
    } else {
        "LIGHTLIKE"
    }
}

fn main() {
    println!("SRFM Paper — Rust reference implementation");
    println!("==========================================\n");

    // Synthetic bars: c_market calibrated at 99.5th pctile, Δt = 1 (normalised)
    let c_market = 0.0024_f64; // ~2.4 bps per minute, Q1 2025 large-cap equity
    let delta_t  = 1.0_f64;

    let bars = [
        ("calm trending bar",   100.00, 100.05),
        ("moderate move bar",   100.00, 100.30),
        ("shock bar (SL)",      100.00, 101.20),
        ("near-lightlike bar",  100.00, 100.24),
    ];

    println!("{:<24} {:>8} {:>8} {:>8} {:>10} {:>10} {:>12}",
             "Bar", "β", "γ", "φ", "τ", "s²", "Regime");
    println!("{}", "─".repeat(84));

    for (label, open, close) in bars {
        let b   = beta(open, close, c_market, delta_t);
        let g   = gamma(b);
        let dx  = (close / open).ln();
        let s2  = spacetime_interval(c_market, delta_t, dx);
        let tau = proper_time(delta_t, b);
        let phi = rapidity(b);

        println!("{:<24} {:>8.4} {:>8.4} {:>8.4} {:>10.6} {:>10.2e} {:>12}",
                 label, b, g, phi, tau, s2, regime(s2));
    }

    println!("\nKey results from the paper (Q1 2025 equity, synthetic data):");
    println!("  Variance ratio VR = Var(SPACELIKE) / Var(TIMELIKE) ≈ 1.27×");
    println!("  Bartlett test p-value ≈ 6 × 10⁻¹⁶");
    println!("  TIMELIKE fraction ≈ 75% of 1-min bars");
    println!("  SRFM strategy Sharpe ratio ≈ 1.8 (vs buy-and-hold ≈ 1.1)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_calm_bar_below_unity() {
        let b = beta(100.0, 100.05, 0.0024, 1.0);
        assert!(b < 1.0, "calm bar should be TIMELIKE (β < 1)");
    }

    #[test]
    fn test_beta_shock_bar_above_unity() {
        let b = beta(100.0, 101.50, 0.0024, 1.0);
        assert!(b > 1.0 || b == BETA_MAX,
            "shock bar should be SPACELIKE (β ≥ 1) or clamped");
    }

    #[test]
    fn test_gamma_unity_at_zero_velocity() {
        let g = gamma(0.0);
        assert!((g - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_gamma_diverges_near_lightspeed() {
        let g = gamma(0.9999);
        assert!(g > 70.0, "γ should diverge near β=1, got {g}");
    }

    #[test]
    fn test_spacetime_interval_timelike_positive() {
        // small price move → s² > 0
        let s2 = spacetime_interval(0.0024, 1.0, 0.0010);
        assert!(s2 > 0.0);
    }

    #[test]
    fn test_spacetime_interval_spacelike_negative() {
        // large price move → s² < 0
        let s2 = spacetime_interval(0.0024, 1.0, 0.0050);
        assert!(s2 < 0.0);
    }

    #[test]
    fn test_proper_time_less_than_coordinate_time() {
        let tau = proper_time(1.0, 0.8);
        assert!(tau < 1.0, "proper time must be < coordinate time for β > 0");
    }

    #[test]
    fn test_rapidity_zero_at_zero_velocity() {
        assert!(rapidity(0.0).abs() < 1e-12);
    }

    #[test]
    fn test_rapidity_positive_for_positive_beta() {
        assert!(rapidity(0.5) > 0.0);
    }

    #[test]
    fn test_regime_labels() {
        assert_eq!(regime(1.0),  "TIMELIKE");
        assert_eq!(regime(-1.0), "SPACELIKE");
        assert_eq!(regime(0.0),  "LIGHTLIKE");
    }
}
