//! Symmetric-RTP destination learning with bounded rebinding.
//!
//! A peer behind NAT may send RTP from a different socket than the address in
//! SDP.  The first valid RTP packet therefore latches the observed source.
//! Moving an established latch is deliberately stricter: the source must pass
//! a short same-SSRC/sequence probation and the number of moves is bounded.
//! This does not replace SRTP authentication, but it prevents a single spoofed
//! UDP packet from redirecting an established plain-RTP call.
//!
//! The peer's signalled address (SDP, or the ICE-nominated pair) may reclaim
//! a learned latch, under the same rules as any other move. Media ports are
//! reused: a call can be handed a port that a previous call — possibly in a
//! previous process, whose peer has not noticed yet — is still receiving a
//! stream on. If that stale stream's first packet wins the initial latch, the
//! new call's own peer was starved: a different IP is rejected outright, and a
//! same-IP candidate never finished probation because every packet from the
//! latched source cancelled it. So, for [`SIGNALLED_RECLAIM_WINDOW`] after the
//! address is signalled, a stream from exactly that address may move the latch
//! even across IPs and SSRCs: it must pass the normal probation (N in-sequence
//! packets with one SSRC within the rebind window), it counts against
//! `max_rebindings`, and packets from the latched source do not cancel it. One
//! spoofed packet from the signalled address cannot move an established latch,
//! and after the window the signalled address has no special standing. A peer
//! behind NAT, or a carrier whose media relay sends from an address other than
//! its SDP `c=`, never sends from the signalled address and is learned as
//! before.

use std::net::SocketAddr;
use std::time::{Duration, Instant};

/// How long after the peer's address is signalled a stream from it may
/// reclaim a latch learned from another source (see the module docs).
pub const SIGNALLED_RECLAIM_WINDOW: Duration = Duration::from_secs(10);

/// Policy for learning and rebinding a symmetric-RTP destination.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SymmetricRtpPolicy {
    /// Whether validated inbound RTP may update the outbound destination.
    pub enabled: bool,
    /// Permit a rebind to a different IP address. The default only permits a
    /// source-port change on the already-latched IP.
    pub allow_ip_change: bool,
    /// Consecutive packets required from a new source before it becomes the
    /// outbound destination.
    pub probation_packets: u8,
    /// Maximum accepted destination changes after the initial latch.
    pub max_rebindings: u8,
    /// Maximum time allowed to complete one probation sequence.
    pub rebind_window: Duration,
    /// Largest forward RTP sequence-number jump accepted during probation.
    pub max_sequence_jump: u16,
}

impl Default for SymmetricRtpPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            allow_ip_change: false,
            probation_packets: 3,
            max_rebindings: 2,
            rebind_window: Duration::from_secs(2),
            max_sequence_jump: 512,
        }
    }
}

impl SymmetricRtpPolicy {
    /// A policy that never learns a destination from inbound RTP.
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            allow_ip_change: false,
            probation_packets: 1,
            max_rebindings: 0,
            rebind_window: Duration::from_secs(1),
            max_sequence_jump: 1,
        }
    }

    /// Validate bounds before a receive task is started.
    pub fn validate(self) -> Result<(), &'static str> {
        if self.probation_packets == 0 || self.probation_packets > 32 {
            return Err("symmetric RTP probation_packets must be between 1 and 32");
        }
        if self.rebind_window.is_zero() || self.rebind_window > Duration::from_secs(60) {
            return Err("symmetric RTP rebind_window must be greater than zero and at most 60s");
        }
        if self.max_sequence_jump == 0 || self.max_sequence_jump >= 0x8000 {
            return Err("symmetric RTP max_sequence_jump must be between 1 and 32767");
        }
        Ok(())
    }

    /// Permit authenticated deployments to opt into bounded IP changes.
    pub const fn with_ip_change(mut self, enabled: bool) -> Self {
        self.allow_ip_change = enabled;
        self
    }
}

/// Aggregate-only diagnostics. No peer address, SSRC, token, or session ID is
/// exposed, so this snapshot is safe to include in operational diagnostics.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SymmetricRtpDiagnostics {
    pub destination_learned: bool,
    pub accepted_rebindings: u64,
    pub probation_packets: u64,
    pub rejected_packets: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SymmetricRtpDecision {
    Accept,
    LatchInitial,
    Probation,
    Rebind,
    Reject,
}

#[derive(Clone, Copy, Debug)]
struct LatchedSource {
    address: SocketAddr,
    ssrc: u32,
    last_sequence: u16,
}

#[derive(Clone, Copy, Debug)]
struct RebindCandidate {
    address: SocketAddr,
    ssrc: u32,
    last_sequence: u16,
    packets: u8,
    started_at: Instant,
    /// Started by a stream from the signalled address (a reclaim): packets
    /// from the latched source do not cancel it.
    reclaim: bool,
}

pub(crate) struct SymmetricRtpLearner {
    policy: SymmetricRtpPolicy,
    /// The peer's signalled RTP address (SDP or ICE) and when it was
    /// signalled, when known.
    signalled: Option<(SocketAddr, Instant)>,
    latched: Option<LatchedSource>,
    candidate: Option<RebindCandidate>,
    rebindings: u8,
}

impl SymmetricRtpLearner {
    pub(crate) fn new(policy: SymmetricRtpPolicy) -> Self {
        Self {
            policy,
            signalled: None,
            latched: None,
            candidate: None,
            rebindings: 0,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.latched = None;
        self.candidate = None;
        self.rebindings = 0;
    }

    /// Record the address the peer signalled (SDP connection address or the
    /// ICE-nominated pair). See the module docs for why it outranks a learned
    /// source.
    pub(crate) fn set_signalled(&mut self, signalled: Option<(SocketAddr, Instant)>) {
        self.signalled = signalled;
    }

    /// Whether a packet from `source` may start or continue a reclaim.
    fn may_reclaim(&self, source: SocketAddr, now: Instant) -> bool {
        self.signalled.is_some_and(|(address, since)| {
            address == source
                && now.saturating_duration_since(since) <= SIGNALLED_RECLAIM_WINDOW
                && self.rebindings < self.policy.max_rebindings
        })
    }

    /// Probation for a stream from the signalled address. The same rules as
    /// a learned move, without the same-IP and same-SSRC-as-latched limits
    /// (the latched source is presumed stale).
    fn observe_reclaim(
        &mut self,
        source: SocketAddr,
        ssrc: u32,
        sequence: u16,
        now: Instant,
    ) -> SymmetricRtpDecision {
        let continuing = self.candidate.filter(|candidate| {
            candidate.reclaim
                && candidate.address == source
                && candidate.ssrc == ssrc
                && now.duration_since(candidate.started_at) <= self.policy.rebind_window
        });
        let candidate = match continuing {
            Some(candidate) => {
                let delta = sequence.wrapping_sub(candidate.last_sequence);
                if delta == 0 || delta > self.policy.max_sequence_jump {
                    self.candidate = None;
                    return SymmetricRtpDecision::Reject;
                }
                RebindCandidate {
                    last_sequence: sequence,
                    packets: candidate.packets.saturating_add(1),
                    ..candidate
                }
            }
            None => RebindCandidate {
                address: source,
                ssrc,
                last_sequence: sequence,
                packets: 1,
                started_at: now,
                reclaim: true,
            },
        };
        if candidate.packets >= self.policy.probation_packets {
            self.accept_candidate(candidate);
            SymmetricRtpDecision::Rebind
        } else {
            self.candidate = Some(candidate);
            SymmetricRtpDecision::Probation
        }
    }

    pub(crate) fn observe(
        &mut self,
        source: SocketAddr,
        ssrc: u32,
        sequence: u16,
        now: Instant,
    ) -> SymmetricRtpDecision {
        if !self.policy.enabled {
            return SymmetricRtpDecision::Accept;
        }

        let Some(mut latched) = self.latched else {
            self.latched = Some(LatchedSource {
                address: source,
                ssrc,
                last_sequence: sequence,
            });
            return SymmetricRtpDecision::LatchInitial;
        };

        if source == latched.address {
            // A packet from the established source wins over an in-progress
            // candidate. Legitimate SSRC changes are learned only on that
            // established tuple.
            latched.ssrc = ssrc;
            latched.last_sequence = sequence;
            self.latched = Some(latched);
            // A reclaim by the signalled peer is not cancelled by the source
            // it is reclaiming from: that source may be a stale stream that
            // never stops.
            if !self.candidate.is_some_and(|candidate| candidate.reclaim) {
                self.candidate = None;
            }
            return SymmetricRtpDecision::Accept;
        }

        if self.may_reclaim(source, now) {
            return self.observe_reclaim(source, ssrc, sequence, now);
        }

        if self.rebindings >= self.policy.max_rebindings
            || (!self.policy.allow_ip_change && source.ip() != latched.address.ip())
            || ssrc != latched.ssrc
        {
            self.candidate = None;
            return SymmetricRtpDecision::Reject;
        }

        let candidate = match self.candidate {
            Some(candidate)
                if !candidate.reclaim
                    && candidate.address == source
                    && candidate.ssrc == ssrc
                    && now.duration_since(candidate.started_at) <= self.policy.rebind_window =>
            {
                candidate
            }
            _ => {
                let delta = sequence.wrapping_sub(latched.last_sequence);
                if delta == 0 || delta > self.policy.max_sequence_jump {
                    self.candidate = None;
                    return SymmetricRtpDecision::Reject;
                }
                let candidate = RebindCandidate {
                    address: source,
                    ssrc,
                    last_sequence: sequence,
                    packets: 1,
                    started_at: now,
                    reclaim: false,
                };
                if self.policy.probation_packets == 1 {
                    self.accept_candidate(candidate);
                    return SymmetricRtpDecision::Rebind;
                }
                self.candidate = Some(candidate);
                return SymmetricRtpDecision::Probation;
            }
        };

        let delta = sequence.wrapping_sub(candidate.last_sequence);
        if delta == 0 || delta > self.policy.max_sequence_jump {
            self.candidate = None;
            return SymmetricRtpDecision::Reject;
        }

        let candidate = RebindCandidate {
            last_sequence: sequence,
            packets: candidate.packets.saturating_add(1),
            ..candidate
        };
        if candidate.packets >= self.policy.probation_packets {
            self.accept_candidate(candidate);
            SymmetricRtpDecision::Rebind
        } else {
            self.candidate = Some(candidate);
            SymmetricRtpDecision::Probation
        }
    }

    fn accept_candidate(&mut self, candidate: RebindCandidate) {
        self.latched = Some(LatchedSource {
            address: candidate.address,
            ssrc: candidate.ssrc,
            last_sequence: candidate.last_sequence,
        });
        self.candidate = None;
        self.rebindings = self.rebindings.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(ip: [u8; 4], port: u16) -> SocketAddr {
        SocketAddr::from((ip, port))
    }

    #[test]
    fn rebind_requires_probation_and_preserves_same_ip_by_default() {
        let now = Instant::now();
        let mut learner = SymmetricRtpLearner::new(SymmetricRtpPolicy::default());
        let original = addr([192, 0, 2, 10], 20_000);
        let rebound = addr([192, 0, 2, 10], 30_000);

        assert_eq!(
            learner.observe(original, 7, 100, now),
            SymmetricRtpDecision::LatchInitial
        );
        assert_eq!(
            learner.observe(rebound, 7, 101, now),
            SymmetricRtpDecision::Probation
        );
        assert_eq!(
            learner.observe(rebound, 7, 102, now),
            SymmetricRtpDecision::Probation
        );
        assert_eq!(
            learner.observe(rebound, 7, 103, now),
            SymmetricRtpDecision::Rebind
        );
        assert_eq!(
            learner.observe(rebound, 7, 104, now),
            SymmetricRtpDecision::Accept
        );
    }

    #[test]
    fn different_ip_ssrc_and_large_jump_are_rejected() {
        let now = Instant::now();
        let mut learner = SymmetricRtpLearner::new(SymmetricRtpPolicy::default());
        let original = addr([192, 0, 2, 10], 20_000);
        assert_eq!(
            learner.observe(original, 7, 100, now),
            SymmetricRtpDecision::LatchInitial
        );
        assert_eq!(
            learner.observe(addr([198, 51, 100, 1], 20_001), 7, 101, now),
            SymmetricRtpDecision::Reject
        );
        assert_eq!(
            learner.observe(addr([192, 0, 2, 10], 20_001), 8, 101, now),
            SymmetricRtpDecision::Reject
        );
        assert_eq!(
            learner.observe(addr([192, 0, 2, 10], 20_001), 7, 10_000, now),
            SymmetricRtpDecision::Reject
        );
    }

    #[test]
    fn established_source_cancels_spoof_candidate() {
        let now = Instant::now();
        let mut learner = SymmetricRtpLearner::new(SymmetricRtpPolicy::default());
        let original = addr([192, 0, 2, 10], 20_000);
        let candidate = addr([192, 0, 2, 10], 20_001);
        learner.observe(original, 7, 100, now);
        assert_eq!(
            learner.observe(candidate, 7, 101, now),
            SymmetricRtpDecision::Probation
        );
        assert_eq!(
            learner.observe(original, 7, 101, now),
            SymmetricRtpDecision::Accept
        );
        assert_eq!(
            learner.observe(candidate, 7, 102, now),
            SymmetricRtpDecision::Probation
        );
    }
    const SIGNALLED: [u8; 4] = [203, 0, 113, 5];

    /// A learner whose peer signalled `203.0.113.5:44276` at `now`, latched
    /// onto a stale stream from `stale` that arrived first.
    fn latched_to_stale(now: Instant, stale: SocketAddr) -> (SymmetricRtpLearner, SocketAddr) {
        let mut learner = SymmetricRtpLearner::new(SymmetricRtpPolicy::default());
        let signalled = addr(SIGNALLED, 44_276);
        learner.set_signalled(Some((signalled, now)));
        assert_eq!(
            learner.observe(stale, 99, 5_000, now),
            SymmetricRtpDecision::LatchInitial
        );
        (learner, signalled)
    }

    #[test]
    fn signalled_peer_reclaims_a_stale_latch_after_probation_despite_the_stale_stream() {
        let now = Instant::now();
        let stale = addr(SIGNALLED, 51_000);
        let (mut learner, signalled) = latched_to_stale(now, stale);
        // The real peer and the stale stream interleave; the stale packets
        // must not cancel the reclaim.
        assert_eq!(
            learner.observe(signalled, 7, 100, now),
            SymmetricRtpDecision::Probation
        );
        assert_eq!(
            learner.observe(stale, 99, 5_001, now),
            SymmetricRtpDecision::Accept
        );
        assert_eq!(
            learner.observe(signalled, 7, 101, now),
            SymmetricRtpDecision::Probation
        );
        assert_eq!(
            learner.observe(stale, 99, 5_002, now),
            SymmetricRtpDecision::Accept
        );
        assert_eq!(
            learner.observe(signalled, 7, 102, now),
            SymmetricRtpDecision::Rebind
        );
        assert_eq!(
            learner.observe(signalled, 7, 103, now),
            SymmetricRtpDecision::Accept
        );
        // The stale stream (other SSRC) cannot take it back.
        assert_eq!(
            learner.observe(stale, 99, 5_003, now),
            SymmetricRtpDecision::Reject
        );
    }

    #[test]
    fn signalled_peer_on_another_ip_reclaims_after_probation() {
        let now = Instant::now();
        let (mut learner, signalled) = latched_to_stale(now, addr([198, 51, 100, 9], 40_000));
        for sequence in 100..102 {
            assert_eq!(
                learner.observe(signalled, 7, sequence, now),
                SymmetricRtpDecision::Probation
            );
        }
        assert_eq!(
            learner.observe(signalled, 7, 102, now),
            SymmetricRtpDecision::Rebind
        );
    }

    #[test]
    fn one_spoofed_packet_from_the_signalled_address_cannot_steal_the_latch() {
        let now = Instant::now();
        // A carrier whose relay sends from another address than its c=.
        let relay = addr([198, 51, 100, 20], 30_000);
        let (mut learner, signalled) = latched_to_stale(now, relay);
        assert_eq!(
            learner.observe(signalled, 1234, 9, now),
            SymmetricRtpDecision::Probation
        );
        for sequence in 5_001..5_010 {
            assert_eq!(
                learner.observe(relay, 99, sequence, now),
                SymmetricRtpDecision::Accept,
                "the relay keeps the latch"
            );
        }
        // Out-of-sequence or other-SSRC follow-ups do not complete it either.
        assert_eq!(
            learner.observe(signalled, 1234, 9, now),
            SymmetricRtpDecision::Reject
        );
        assert_eq!(
            learner.observe(signalled, 4321, 10, now),
            SymmetricRtpDecision::Probation
        );
        assert_eq!(
            learner.observe(relay, 99, 5_010, now),
            SymmetricRtpDecision::Accept
        );
    }

    #[test]
    fn a_relay_sending_from_another_address_than_c_latches_and_stays() {
        let now = Instant::now();
        let mut learner = SymmetricRtpLearner::new(SymmetricRtpPolicy::default());
        learner.set_signalled(Some((addr(SIGNALLED, 44_276), now)));
        let relay = addr([198, 51, 100, 20], 30_000);
        assert_eq!(
            learner.observe(relay, 7, 100, now),
            SymmetricRtpDecision::LatchInitial
        );
        for sequence in 101..200 {
            assert_eq!(
                learner.observe(relay, 7, sequence, now),
                SymmetricRtpDecision::Accept
            );
        }
    }

    #[test]
    fn the_signalled_address_has_no_standing_after_the_window() {
        let now = Instant::now();
        let stale = addr(SIGNALLED, 51_000);
        let (mut learner, signalled) = latched_to_stale(now, stale);
        let late = now + SIGNALLED_RECLAIM_WINDOW + Duration::from_millis(1);
        // Same IP, other SSRC than the latch: an ordinary move, refused.
        assert_eq!(
            learner.observe(signalled, 7, 100, late),
            SymmetricRtpDecision::Reject
        );
    }

    #[test]
    fn a_reclaim_counts_against_max_rebindings() {
        let now = Instant::now();
        let policy = SymmetricRtpPolicy {
            max_rebindings: 1,
            ..SymmetricRtpPolicy::default()
        };
        let mut learner = SymmetricRtpLearner::new(policy);
        let signalled = addr(SIGNALLED, 44_276);
        learner.set_signalled(Some((signalled, now)));
        let stale = addr(SIGNALLED, 51_000);
        learner.observe(stale, 99, 5_000, now);
        for sequence in 100..103 {
            learner.observe(signalled, 7, sequence, now);
        }
        assert_eq!(learner.rebindings, 1);
        // The budget is spent: even a same-SSRC, same-IP move is refused now.
        let other_port = addr(SIGNALLED, 44_300);
        assert_eq!(
            learner.observe(other_port, 7, 103, now),
            SymmetricRtpDecision::Reject
        );
        // And a second reclaim window does not reopen it.
        learner.set_signalled(Some((other_port, now)));
        assert_eq!(
            learner.observe(other_port, 7, 104, now),
            SymmetricRtpDecision::Reject
        );
    }

    #[test]
    fn nat_peer_that_never_uses_its_signalled_address_is_learned_as_before() {
        let now = Instant::now();
        let mut learner = SymmetricRtpLearner::new(SymmetricRtpPolicy::default());
        // A phone behind NAT signals its private address.
        learner.set_signalled(Some((addr([10, 0, 0, 7], 4_000), now)));
        let public = addr([192, 0, 2, 10], 20_000);
        assert_eq!(
            learner.observe(public, 7, 100, now),
            SymmetricRtpDecision::LatchInitial
        );
        assert_eq!(
            learner.observe(public, 7, 101, now),
            SymmetricRtpDecision::Accept
        );
        assert_eq!(
            learner.observe(addr([198, 51, 100, 1], 20_001), 7, 102, now),
            SymmetricRtpDecision::Reject
        );
    }
}
