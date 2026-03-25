//! Entity types: SpanLabel, RiskLevel, DetectionResult, Tier1Label, Tier1Finding.
//!
//! Spec: specifications/entities.md

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

/// Tier 2 quasi-identifier categories (entities.md §1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING-KEBAB-CASE")]
pub enum SpanLabel {
    Location,
    Workplace,
    Routine,
    #[serde(rename = "MEDICAL-CONTEXT")]
    MedicalContext,
    Demographic,
    #[serde(rename = "DEVICE-ID")]
    DeviceId,
    Credential,
    #[serde(rename = "QUASI-ID")]
    QuasiId,
}

impl fmt::Display for SpanLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Location => write!(f, "LOCATION"),
            Self::Workplace => write!(f, "WORKPLACE"),
            Self::Routine => write!(f, "ROUTINE"),
            Self::MedicalContext => write!(f, "MEDICAL-CONTEXT"),
            Self::Demographic => write!(f, "DEMOGRAPHIC"),
            Self::DeviceId => write!(f, "DEVICE-ID"),
            Self::Credential => write!(f, "CREDENTIAL"),
            Self::QuasiId => write!(f, "QUASI-ID"),
        }
    }
}

impl SpanLabel {
    /// Parse from string, returning None for invalid values.
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "LOCATION" => Some(Self::Location),
            "WORKPLACE" => Some(Self::Workplace),
            "ROUTINE" => Some(Self::Routine),
            "MEDICAL-CONTEXT" => Some(Self::MedicalContext),
            "DEMOGRAPHIC" => Some(Self::Demographic),
            "DEVICE-ID" => Some(Self::DeviceId),
            "CREDENTIAL" => Some(Self::Credential),
            "QUASI-ID" => Some(Self::QuasiId),
            _ => None,
        }
    }
}

/// Risk level enumeration (entities.md §2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

impl fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "LOW"),
            Self::Medium => write!(f, "MEDIUM"),
            Self::High => write!(f, "HIGH"),
        }
    }
}

impl RiskLevel {
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "LOW" => Some(Self::Low),
            "MEDIUM" => Some(Self::Medium),
            "HIGH" => Some(Self::High),
            _ => None,
        }
    }
}

/// Detection result for Tier 2 quasi-identifier scan (entities.md §3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetectionResult {
    pub labels: BTreeSet<SpanLabel>,
    pub risk: RiskLevel,
    pub rationale: String,
}

impl DetectionResult {
    /// Construct a DetectionResult, enforcing invariants from entities.md §3.
    ///
    /// Panics if invariants are violated. Use `new_enforced` for auto-correction.
    pub fn new(labels: BTreeSet<SpanLabel>, risk: RiskLevel, rationale: String) -> Self {
        let result = Self { labels, risk, rationale };
        result.assert_invariants();
        result
    }

    /// Construct a DetectionResult, auto-correcting to satisfy invariants.
    pub fn new_enforced(labels: BTreeSet<SpanLabel>, risk: RiskLevel, rationale: String) -> Self {
        let mut risk = risk;
        let mut rationale = rationale;

        if labels.is_empty() {
            risk = RiskLevel::Low;
        }
        if risk == RiskLevel::Low {
            rationale = String::new();
        }
        if (risk == RiskLevel::Medium || risk == RiskLevel::High)
            && labels.len() >= 2
            && rationale.is_empty()
        {
            rationale = "Multiple quasi-identifiers detected.".to_string();
        }

        Self { labels, risk, rationale }
    }

    /// The empty/clean detection result.
    pub fn empty() -> Self {
        Self {
            labels: BTreeSet::new(),
            risk: RiskLevel::Low,
            rationale: String::new(),
        }
    }

    fn assert_invariants(&self) {
        if self.labels.is_empty() {
            assert_eq!(self.risk, RiskLevel::Low, "empty labels must have LOW risk");
        }
        if self.risk == RiskLevel::Low {
            assert!(self.rationale.is_empty(), "LOW risk must have empty rationale");
        }
        if (self.risk == RiskLevel::Medium || self.risk == RiskLevel::High)
            && self.labels.len() >= 2
        {
            assert!(!self.rationale.is_empty(), "MEDIUM/HIGH with 2+ labels must have rationale");
        }
    }

    /// Serialize to JSON for stderr output. Labels are sorted alphabetically.
    pub fn to_json(&self) -> String {
        let mut labels: Vec<String> = self.labels.iter().map(|l| l.to_string()).collect();
        labels.sort();
        let obj = serde_json::json!({
            "labels": labels,
            "risk": self.risk.to_string(),
            "rationale": self.rationale,
        });
        serde_json::to_string(&obj).expect("DetectionResult serialization cannot fail")
    }
}

/// Tier 1 direct identifier categories (entities.md §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING-KEBAB-CASE")]
pub enum Tier1Label {
    #[serde(rename = "NAME")]
    Name,
    #[serde(rename = "EMAIL")]
    Email,
    #[serde(rename = "PHONE")]
    Phone,
    #[serde(rename = "ADDRESS")]
    Address,
    #[serde(rename = "GOV-ID")]
    GovId,
    #[serde(rename = "FINANCIAL")]
    Financial,
    #[serde(rename = "DOB")]
    Dob,
    #[serde(rename = "BIOMETRIC")]
    Biometric,
}

impl fmt::Display for Tier1Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name => write!(f, "NAME"),
            Self::Email => write!(f, "EMAIL"),
            Self::Phone => write!(f, "PHONE"),
            Self::Address => write!(f, "ADDRESS"),
            Self::GovId => write!(f, "GOV-ID"),
            Self::Financial => write!(f, "FINANCIAL"),
            Self::Dob => write!(f, "DOB"),
            Self::Biometric => write!(f, "BIOMETRIC"),
        }
    }
}

/// A single Tier 1 span-level finding (entities.md §5).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tier1Finding {
    pub label: Tier1Label,
    pub text: String,
    pub start: usize,
    pub end: usize,
}

impl Tier1Finding {
    pub fn new(label: Tier1Label, text: String, start: usize, end: usize) -> Self {
        assert!(!text.is_empty(), "Tier1Finding text must be non-empty");
        assert!(end > start, "Tier1Finding end must be > start");
        Self { label, text, start, end }
    }
}

/// Wrapper for Tier 1 stderr output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier1Output {
    pub findings: Vec<Tier1Finding>,
}

impl Tier1Output {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("Tier1Output serialization cannot fail")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- SpanLabel tests ---

    #[test]
    fn span_label_display_all_values() {
        assert_eq!(SpanLabel::Location.to_string(), "LOCATION");
        assert_eq!(SpanLabel::Workplace.to_string(), "WORKPLACE");
        assert_eq!(SpanLabel::Routine.to_string(), "ROUTINE");
        assert_eq!(SpanLabel::MedicalContext.to_string(), "MEDICAL-CONTEXT");
        assert_eq!(SpanLabel::Demographic.to_string(), "DEMOGRAPHIC");
        assert_eq!(SpanLabel::DeviceId.to_string(), "DEVICE-ID");
        assert_eq!(SpanLabel::Credential.to_string(), "CREDENTIAL");
        assert_eq!(SpanLabel::QuasiId.to_string(), "QUASI-ID");
    }

    #[test]
    fn span_label_from_str_valid() {
        assert_eq!(SpanLabel::from_str_opt("LOCATION"), Some(SpanLabel::Location));
        assert_eq!(SpanLabel::from_str_opt("QUASI-ID"), Some(SpanLabel::QuasiId));
    }

    #[test]
    fn span_label_from_str_invalid() {
        assert_eq!(SpanLabel::from_str_opt("INVALID"), None);
        assert_eq!(SpanLabel::from_str_opt("location"), None);
        assert_eq!(SpanLabel::from_str_opt("NAME"), None); // Tier 1, not Tier 2
    }

    #[test]
    fn span_label_no_overlap_with_tier1() {
        let tier1_values = ["NAME", "EMAIL", "PHONE", "ADDRESS", "GOV-ID", "FINANCIAL", "DOB", "BIOMETRIC"];
        for v in tier1_values {
            assert!(SpanLabel::from_str_opt(v).is_none(), "SpanLabel must not overlap with Tier1Label: {v}");
        }
    }

    // --- RiskLevel tests ---

    #[test]
    fn risk_level_ordering() {
        assert!(RiskLevel::Low < RiskLevel::Medium);
        assert!(RiskLevel::Medium < RiskLevel::High);
    }

    #[test]
    fn risk_level_from_str() {
        assert_eq!(RiskLevel::from_str_opt("LOW"), Some(RiskLevel::Low));
        assert_eq!(RiskLevel::from_str_opt("MEDIUM"), Some(RiskLevel::Medium));
        assert_eq!(RiskLevel::from_str_opt("HIGH"), Some(RiskLevel::High));
        assert_eq!(RiskLevel::from_str_opt("low"), None);
    }

    // --- DetectionResult tests ---

    #[test]
    fn detection_result_empty() {
        let r = DetectionResult::empty();
        assert!(r.labels.is_empty());
        assert_eq!(r.risk, RiskLevel::Low);
        assert!(r.rationale.is_empty());
    }

    #[test]
    fn detection_result_valid_construction() {
        let mut labels = BTreeSet::new();
        labels.insert(SpanLabel::Workplace);
        labels.insert(SpanLabel::Routine);
        let r = DetectionResult::new(labels, RiskLevel::Medium, "workplace + routine".into());
        assert_eq!(r.labels.len(), 2);
        assert_eq!(r.risk, RiskLevel::Medium);
    }

    #[test]
    #[should_panic(expected = "empty labels must have LOW risk")]
    fn detection_result_empty_labels_non_low_panics() {
        DetectionResult::new(BTreeSet::new(), RiskLevel::High, "bad".into());
    }

    #[test]
    #[should_panic(expected = "LOW risk must have empty rationale")]
    fn detection_result_low_with_rationale_panics() {
        let mut labels = BTreeSet::new();
        labels.insert(SpanLabel::Location);
        DetectionResult::new(labels, RiskLevel::Low, "should be empty".into());
    }

    #[test]
    #[should_panic(expected = "MEDIUM/HIGH with 2+ labels must have rationale")]
    fn detection_result_multi_label_no_rationale_panics() {
        let mut labels = BTreeSet::new();
        labels.insert(SpanLabel::Workplace);
        labels.insert(SpanLabel::Routine);
        DetectionResult::new(labels, RiskLevel::High, String::new());
    }

    #[test]
    fn detection_result_enforced_fixes_invariants() {
        // Empty labels with HIGH risk → corrected to LOW
        let r = DetectionResult::new_enforced(BTreeSet::new(), RiskLevel::High, "nope".into());
        assert_eq!(r.risk, RiskLevel::Low);
        assert!(r.rationale.is_empty());

        // Multi-label MEDIUM with no rationale → auto-generated
        let mut labels = BTreeSet::new();
        labels.insert(SpanLabel::Workplace);
        labels.insert(SpanLabel::Routine);
        let r = DetectionResult::new_enforced(labels, RiskLevel::Medium, String::new());
        assert_eq!(r.rationale, "Multiple quasi-identifiers detected.");
    }

    #[test]
    fn detection_result_to_json() {
        let mut labels = BTreeSet::new();
        labels.insert(SpanLabel::Workplace);
        labels.insert(SpanLabel::Routine);
        let r = DetectionResult::new(labels, RiskLevel::Medium, "work + routine".into());
        let json = r.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["risk"], "MEDIUM");
        let label_arr = parsed["labels"].as_array().unwrap();
        assert!(label_arr.contains(&serde_json::json!("WORKPLACE")));
        assert!(label_arr.contains(&serde_json::json!("ROUTINE")));
    }

    #[test]
    fn detection_result_json_labels_sorted() {
        let mut labels = BTreeSet::new();
        labels.insert(SpanLabel::Routine);
        labels.insert(SpanLabel::Demographic);
        labels.insert(SpanLabel::Location);
        let r = DetectionResult::new(labels, RiskLevel::High, "combination".into());
        let json = r.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr: Vec<String> = parsed["labels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        let mut sorted = arr.clone();
        sorted.sort();
        assert_eq!(arr, sorted, "labels must be sorted in JSON output");
    }

    // --- Tier1Label tests ---

    #[test]
    fn tier1_label_display() {
        assert_eq!(Tier1Label::Name.to_string(), "NAME");
        assert_eq!(Tier1Label::GovId.to_string(), "GOV-ID");
    }

    // --- Tier1Finding tests ---

    #[test]
    fn tier1_finding_valid() {
        let f = Tier1Finding::new(Tier1Label::Email, "john@acme.com".into(), 10, 23);
        assert_eq!(f.label, Tier1Label::Email);
        assert_eq!(f.start, 10);
        assert_eq!(f.end, 23);
    }

    #[test]
    #[should_panic(expected = "text must be non-empty")]
    fn tier1_finding_empty_text_panics() {
        Tier1Finding::new(Tier1Label::Email, String::new(), 0, 1);
    }

    #[test]
    #[should_panic(expected = "end must be > start")]
    fn tier1_finding_invalid_offsets_panics() {
        Tier1Finding::new(Tier1Label::Email, "x".into(), 5, 5);
    }

    // --- Tier1Output tests ---

    #[test]
    fn tier1_output_json() {
        let output = Tier1Output {
            findings: vec![
                Tier1Finding::new(Tier1Label::Email, "john@acme.com".into(), 42, 55),
            ],
        };
        let json = output.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["findings"][0]["label"], "EMAIL");
        assert_eq!(parsed["findings"][0]["start"], 42);
    }

    #[test]
    fn tier1_output_empty_findings_json() {
        let output = Tier1Output { findings: vec![] };
        let json = output.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed["findings"].as_array().unwrap().is_empty());
    }
}
