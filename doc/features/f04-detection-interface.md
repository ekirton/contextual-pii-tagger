# F-04: Detection Interface

**Priority:** P0
**Requirements:** R-API-01, R-DET-04

## What This Feature Does

Provides a programmatic interface that application developers use to analyze text for quasi-identifier PII. A developer loads the model once, then passes text strings to the interface and receives structured results — the set of quasi-identifier categories present, a risk score, and a rationale.

The interface operates entirely offline. Once the model is loaded, no network calls are made during detection. All inference happens locally on the user's machine.

## Why It Exists

The model is only useful if developers can integrate it into their applications. The detection interface is the primary way users interact with the product — whether they are building an LLM pipeline pre-filter, a chat audit tool, or a data anonymization workflow.

Offline operation is essential because the product's purpose is to prevent PII from leaving the local environment. Making network calls during detection would defeat that purpose.

## Design Tradeoffs

- The interface is a code-level API, not a GUI or CLI tool. This targets the primary user segment (developers integrating into applications) at the cost of accessibility for non-developers.
- The interface returns structured data (category labels, risk score, rationale) rather than raw model output. This means the interface includes output parsing, which adds a layer of complexity but makes the results immediately usable.
- The model must be small enough to load and run on consumer hardware without a GPU. This constrains model size but ensures broad accessibility.

## What This Feature Does Not Provide

- A graphical user interface.
- A command-line tool (though one could be built on top of this interface).
- Network-based inference or cloud deployment (see F-12: REST API for that).
- Tier 1 direct identifier detection (delegated to existing tools).

## Acceptance Criteria

### AC-01: Load and detect
**GIVEN** a developer has installed the package and downloaded the model
**WHEN** they load the model and pass a text string to the detection interface
**THEN** the interface returns structured results containing the quasi-identifier categories present, a risk score, and a rationale
*(Traces to R-API-01)*

### AC-02: Offline operation
**GIVEN** the model has been loaded
**WHEN** detection is performed on one or more texts
**THEN** no network calls are made at any point during inference
*(Traces to R-DET-04)*

### AC-03: Runs on consumer hardware
**GIVEN** a machine with no specialized GPU
**WHEN** the model is loaded and detection is performed
**THEN** inference completes successfully at a speed usable for interactive workflows

### AC-04: Structured output format
**GIVEN** text containing quasi-identifier PII
**WHEN** the text is analyzed
**THEN** the result contains: the set of quasi-identifier categories present, an overall risk score (LOW/MEDIUM/HIGH), and a rationale string
