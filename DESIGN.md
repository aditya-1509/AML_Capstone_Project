---
name: Heartlens
description: A warm workspace for submitted predictions and model evidence.
colors:
  cream: "#FCF6ED"
  paper: "#F1E5D7"
  hero: "#EED7C2"
  ink: "#3B261E"
  rust: "#A84E32"
  muted: "#765C4E"
  line: "#D8C5B2"
  supporting-ink: "#614333"
  metric-surface: "#FFFDF9"
  sidebar-copy: "#E5CDB7"
  sidebar-hover: "#654535"
  chart-brown: "#765238"
  chart-ochre: "#BC783B"
  chart-sage: "#526955"
  chart-mauve: "#9A6170"
  chart-grid: "#E5D7C7"
  chart-subdued: "#C8A68B"
typography:
  display:
    fontFamily: "DM Sans, sans-serif"
    fontSize: "clamp(2rem, 3.2vw, 2.8rem)"
    fontWeight: 750
    lineHeight: 1.15
    letterSpacing: "-0.025em"
  headline:
    fontFamily: "DM Sans, sans-serif"
    fontSize: "1.6rem"
    fontWeight: 750
    letterSpacing: "-0.025em"
  title:
    fontFamily: "DM Sans, sans-serif"
    fontSize: "1.15rem"
    fontWeight: 750
    letterSpacing: "-0.025em"
  subtitle:
    fontFamily: "DM Sans, sans-serif"
    fontSize: "1rem"
    fontWeight: 750
    letterSpacing: "-0.025em"
  body:
    fontFamily: "DM Sans, sans-serif"
    lineHeight: 1.65
  sidebar-brand:
    fontFamily: "DM Sans, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 800
    letterSpacing: "-0.03em"
  score:
    fontFamily: "DM Sans, sans-serif"
    fontSize: "clamp(3rem, 5vw, 4.5rem)"
    fontWeight: 750
    lineHeight: 1.1
    letterSpacing: "-0.035em"
  metric:
    fontFamily: "DM Sans, sans-serif"
    fontSize: "2rem"
    fontWeight: 750
  label:
    fontFamily: "DM Sans, sans-serif"
    fontSize: "0.85rem"
rounded:
  hero: "14px"
  panel: "12px"
  notice: "10px"
  navigation: "8px"
spacing:
  desktop-gutter: "2rem"
  mobile-gutter: "1rem"
  sidebar-gutter: "1.3rem"
  mobile-gap: "1rem"
components:
  sidebar:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.cream}"
    padding: "2rem 1.3rem"
  sidebar-link:
    textColor: "{colors.cream}"
    rounded: "{rounded.navigation}"
    padding: "0.65rem 0.8rem"
  sidebar-link-active:
    backgroundColor: "{colors.chart-brown}"
    textColor: "{colors.cream}"
    rounded: "{rounded.navigation}"
  sidebar-link-hover:
    backgroundColor: "{colors.sidebar-hover}"
    textColor: "{colors.cream}"
    rounded: "{rounded.navigation}"
  page-hero:
    backgroundColor: "{colors.hero}"
    textColor: "{colors.ink}"
    rounded: "{rounded.hero}"
    padding: "2.2rem 2rem 1.5rem"
  native-field:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
  result-panel:
    rounded: "{rounded.panel}"
    textColor: "{colors.ink}"
  model-score:
    textColor: "{colors.rust}"
    typography: "{typography.score}"
    padding: "0.4rem 0"
  metric:
    backgroundColor: "{colors.metric-surface}"
    rounded: "{rounded.panel}"
    typography: "{typography.metric}"
  notice:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.supporting-ink}"
    rounded: "{rounded.notice}"
  mobile-result:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.cream}"
    padding: "0.8rem 1.1rem"
---

# Design System: Heartlens

## Overview

**Creative North Star: "The Warm Study Workspace"**

Heartlens presents study evidence in a warm, practical workspace. A dark brown sidebar frames a centered cream content column, a peach banner introduces each page, and bold DM Sans creates a clear hierarchy. Terracotta carries model emphasis; native Streamlit controls remain recognizable.

Prediction and Model studio are the two navigation destinations. Forms, submitted results and evidence follow a readable vertical sequence. Fine borders and tonal fills distinguish sections; compact metric cards support the evidence without competing with the main score.

**Key Characteristics:**

- Warm cream canvas with a dark brown sidebar.
- Bold sans-serif headings and legible native controls.
- Centered banner, two-column profile inputs and sequential evidence.
- Flat bordered surfaces with explicit prediction submission.

## Colors

The frontmatter owns the exact values. The palette preserves cream, warm brown and terracotta while adding a peach banner and pale metric surfaces.

### Primary

- **Terracotta:** active controls, the submitted score, the selected model and the first categorical chart series.

### Secondary

- **Warm brown:** active sidebar links and chart differentiation.
- **Ochre:** keyboard focus and chart differentiation.
- **Sage and mauve:** additional model series.
- **Subdued tan:** comparison bars when random forest receives the emphasis.

### Neutral

- **Cream / paper:** canvas and native control or notice surfaces.
- **Peach hero:** page introduction; also the text-selection background.
- **Brown ink / supporting ink / muted brown:** primary reading, banner or notice copy, and captions respectively.
- **Metric surface:** nearly white metric cards.
- **Warm line / chart grid:** container boundaries, rules and quiet plot grids.
- **Sidebar copy / sidebar hover:** pale secondary copy and the raised tonal hover state against the brown sidebar.

**The Warm Contrast Rule.** Use cream on the brown sidebar, brown ink on pale surfaces, and terracotta for model emphasis.

## Typography

Use self-hosted variable DM Sans throughout, with a sans-serif fallback and `font-display: swap`. The extracted display, headline, title and subtitle roles map to h1–h4. Bold headings use balanced wrapping and moderately tight tracking. Body and native-control sizes remain framework-managed.

The submitted score is bold terracotta; metric values use the smaller metric role. Metric labels and score captions share the label size. Sidebar descriptive copy uses (0.74rem), with line heights (1.5) for the introduction and (1.65) for notes. Notice copy is (0.88rem); Plotly text is (13px). At the mobile breakpoint, h1 becomes (2rem).

**The Readable Caption Rule.** Keep captions fully opaque in muted brown; supporting evidence must remain readable.

## Layout

A centered main container has maximum width (960px) and padding (3.8rem 2rem 2rem). The persistent desktop sidebar is constrained to (245–260px), with content padding (2rem 1.3rem). The peach banner spans the content column; its description is limited to (65ch). Text remains left aligned inside the centered banner.

Prediction flows from banner and notice to a bordered profile panel. A full-width example selector precedes two native input columns. The result appears below the form, with score and explanation in relative columns (1:2), followed by comparison chart, table and supporting explanation. Model studio orders selection, retrospective performance and threshold exploration vertically; related metric and plot groups use native columns.

At (750px), main padding becomes (4rem 1rem 7rem), banner padding becomes (1.5rem 1.15rem 1rem), and horizontal gaps become (1rem). Streamlit manages column collapse and sidebar reveal. Preserve top clearance below framework chrome and bottom clearance above the fixed last-prediction strip.

## Elevation & Depth

Use flat tonal fill, thin warm borders and rounded containers. Notices have no border. Metric cards use a pale surface and native borders where requested. The sole authored shadow (`0 -2px 12px #3B261E20`) separates the fixed mobile result strip from scrolling content.

**The Flat Surface Rule.** Separate sections with tonal fill and fine borders. Reserve the authored shadow for the fixed mobile score strip.

## Shapes

Use the hero, panel, notice and navigation radii in the frontmatter. The mobile hero uses the panel radius. Native inputs, toggles, steppers, sliders, expanders and tables retain Streamlit-managed geometry; do not infer a separate control-radius scale.

## Components

### Sidebar navigation

A bold HeartLens brand and pale study descriptor precede exactly two links: Prediction and Model studio. Cream link text sits against brown; active links use warm brown and hover uses the sidebar-hover token. Notes remain small and pale. Link padding and radius are authored; native navigation semantics and responsive behavior remain intact.

### Page banner

A peach rounded container holds one bold title and a short description in supporting ink. It introduces both pages without illustration or extra eyebrow labels. Its mobile padding and radius reduce with the content column.

### Inputs / Fields

The profile panel groups native number inputs and selectboxes into two columns. Paper surfaces, brown text and terracotta accents match the light theme. Labels, units, help and optional missing values stay visible. A native Streamlit form buffers edits; its full-width Predict button submits all current values together. Enter-to-submit is disabled. Typing alone never runs inference. Native hover, disabled and validation states remain framework-managed. Authored keyboard focus is an ochre outline (3px) with offset (3px); the input caret is terracotta.

### Prediction result

The bordered result panel pairs a prominent score with cutoff interpretation, a limitation caption and a compact horizontal plot. Brown dashed lines identify the study cutoff. The missing-input caption and model-comparison chart follow below. Plots use transparent backgrounds, warm grid lines and the observed categorical colors.

### Metrics and notices

Metrics use bold DM Sans values, muted labels, pale surfaces and the panel radius. Group them according to the evidence: two selection metrics, three retrospective metrics, and four threshold metrics. Notices use the paper fill, notice radius, supporting ink and native internal padding.

### Mobile result strip

The fixed bottom strip shows the model, score and cutoff relation in a two-column grid. Cream text sits on ink; the score is (1.8rem), with label (0.8rem) and supporting line (0.7rem). Its padding is defined in the component token. Production positioning is fixed across the viewport with z-index (990).

No custom animation vocabulary is established. Under `prefers-reduced-motion: reduce`, authored transitions and smooth scrolling are disabled.

## Do's and Don'ts

### Do:

- **Do** use DM Sans for headings, body copy, controls and scores.
- **Do** keep navigation limited to Prediction and Model studio.
- **Do** preserve the dark sidebar, warm banner and vertically ordered evidence.
- **Do** retain native control behavior and clear field labels.
- **Do** run inference only after Predict is clicked, and label retained results as the last submitted prediction.
- **Do** preserve mobile clearance for the framework chrome and fixed result strip.

### Don't:

- **Don't** restore serif headings, top navigation or the removed hero illustration.
- **Don't** run inference while the user is editing.
- **Don't** fade captions with reduced opacity.
- **Don't** invent additional button variants or decorative effects.
- **Don't** describe a model score as a diagnosis or validated personal probability.
