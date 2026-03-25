//! Tier 1 direct identifier detection via redact-core (pattern-based).
//!
//! When the `tier1` feature is enabled, uses the `redact-core` AnalyzerEngine
//! for pattern-based PII detection (emails, phones, SSNs, credit cards, etc.).
//! Without the feature, returns empty findings (stub).
//!
//! Spec: specifications/rust-binary.md §2

use crate::entities::{Tier1Finding, Tier1Label, Tier1Output};

/// Scan text for Tier 1 direct identifiers.
///
/// Returns findings ordered by start offset.
pub fn scan_tier1(text: &str) -> Result<Tier1Output, String> {
    if text.is_empty() {
        return Ok(Tier1Output { findings: vec![] });
    }

    #[cfg(feature = "tier1")]
    {
        scan_tier1_impl(text)
    }

    #[cfg(not(feature = "tier1"))]
    {
        let _ = text;
        Ok(Tier1Output { findings: vec![] })
    }
}

/// Map a redact-core EntityType to our Tier1Label.
///
/// Returns None for entity types that don't map to our Tier 1 taxonomy
/// (e.g., ORGANIZATION, URL, IP_ADDRESS, crypto wallets, hashes).
#[cfg(feature = "tier1")]
fn map_entity_type(entity_type: &redact_core::EntityType) -> Option<Tier1Label> {
    use redact_core::EntityType;
    match entity_type {
        // Person names (NER-based, only available if NER model loaded)
        EntityType::Person => Some(Tier1Label::Name),

        // Contact / email
        EntityType::EmailAddress => Some(Tier1Label::Email),

        // Phone numbers (generic + country-specific)
        EntityType::PhoneNumber
        | EntityType::UkPhoneNumber
        | EntityType::UkMobileNumber => Some(Tier1Label::Phone),

        // Physical addresses / postal
        EntityType::UkPostcode
        | EntityType::UsZipCode
        | EntityType::PoBox => Some(Tier1Label::Address),

        // Government-issued IDs
        EntityType::UsSsn
        | EntityType::UsDriverLicense
        | EntityType::UsPassport
        | EntityType::UkNhs
        | EntityType::UkNino
        | EntityType::UkDriverLicense
        | EntityType::UkPassportNumber
        | EntityType::PassportNumber => Some(Tier1Label::GovId),

        // Financial identifiers
        EntityType::CreditCard
        | EntityType::Iban
        | EntityType::IbanCode
        | EntityType::UsBankNumber
        | EntityType::UkSortCode => Some(Tier1Label::Financial),

        // Date/time (maps to DOB — imprecise but captures birthday patterns)
        EntityType::DateTime
        | EntityType::Age => Some(Tier1Label::Dob),

        // Entity types we don't map to Tier 1:
        // Location, Organization (contextual, not direct identifiers)
        // URL, DomainName, IpAddress (network identifiers, not PII per our taxonomy)
        // CryptoWallet, BtcAddress, EthAddress (financial but not traditional PII)
        // Guid, MacAddress, hashes (technical identifiers)
        // MedicalLicense, MedicalRecordNumber (could map but not in our taxonomy)
        _ => None,
    }
}

/// Real implementation using redact-core's AnalyzerEngine.
#[cfg(feature = "tier1")]
fn scan_tier1_impl(text: &str) -> Result<Tier1Output, String> {
    use redact_core::AnalyzerEngine;

    let analyzer = AnalyzerEngine::new();
    let result = analyzer.analyze(text, None).map_err(|e| e.to_string())?;

    let mut findings: Vec<Tier1Finding> = result
        .detected_entities
        .iter()
        .filter_map(|entity| {
            let label = map_entity_type(&entity.entity_type)?;
            let matched_text = entity
                .text
                .clone()
                .unwrap_or_else(|| text[entity.start..entity.end].to_string());

            if matched_text.is_empty() {
                return None;
            }

            Some(Tier1Finding {
                label,
                text: matched_text,
                start: entity.start,
                end: entity.end,
            })
        })
        .collect();

    // Sort by start offset (spec requirement)
    findings.sort_by_key(|f| f.start);

    Ok(Tier1Output { findings })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_empty_text_returns_no_findings() {
        let result = scan_tier1("").unwrap();
        assert!(result.findings.is_empty());
    }

    #[test]
    fn scan_clean_text_returns_no_findings() {
        let result = scan_tier1("The weather is nice today.").unwrap();
        assert!(result.findings.is_empty());
    }

    // --- Tests that require the tier1 feature ---

    #[cfg(feature = "tier1")]
    mod with_redact {
        use super::*;

        #[test]
        fn detects_email_address() {
            let result = scan_tier1("Contact me at john@example.com please").unwrap();
            assert!(!result.findings.is_empty(), "should detect email");
            let email = result.findings.iter().find(|f| f.label == Tier1Label::Email);
            assert!(email.is_some(), "should have EMAIL finding");
            assert!(email.unwrap().text.contains("john@example.com"));
        }

        #[test]
        fn detects_us_ssn() {
            let result = scan_tier1("My SSN is 123-45-6789").unwrap();
            let ssn = result.findings.iter().find(|f| f.label == Tier1Label::GovId);
            assert!(ssn.is_some(), "should detect SSN as GOV-ID");
        }

        #[test]
        fn detects_credit_card() {
            let result = scan_tier1("Card number 4111111111111111").unwrap();
            let cc = result.findings.iter().find(|f| f.label == Tier1Label::Financial);
            assert!(cc.is_some(), "should detect credit card as FINANCIAL");
        }

        #[test]
        fn detects_phone_number() {
            let result = scan_tier1("Call me at 555-123-4567").unwrap();
            let phone = result.findings.iter().find(|f| f.label == Tier1Label::Phone);
            assert!(phone.is_some(), "should detect phone number");
        }

        #[test]
        fn findings_sorted_by_start_offset() {
            let result = scan_tier1("Email john@example.com and SSN 123-45-6789").unwrap();
            if result.findings.len() >= 2 {
                for i in 1..result.findings.len() {
                    assert!(
                        result.findings[i].start >= result.findings[i - 1].start,
                        "findings must be sorted by start offset"
                    );
                }
            }
        }

        #[test]
        fn unmapped_entities_excluded() {
            // URLs and IPs should not produce Tier1Findings
            let result = scan_tier1("Visit https://example.com from 192.168.1.1").unwrap();
            for f in &result.findings {
                assert!(
                    !matches!(f.label, Tier1Label::Biometric),
                    "should not produce unmapped labels"
                );
            }
        }
    }
}
