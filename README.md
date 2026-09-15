# KBDC-WWTP

**Knowledge-Embedded Baseline Dosing Control for risk-constrained external-carbon optimization in wastewater treatment**

Research code associated with the manuscript:

**Knowledge-embedded learning of conservative human dosing baselines for risk-constrained carbon optimization in wastewater nitrogen removal**

KBDC is a knowledge-embedded and risk-constrained framework for external-carbon
dosing in wastewater treatment plants (WWTPs). Instead of treating historical
operator dosing as an optimal carbon-demand target, KBDC interprets historical
dosing as a conservative operational baseline that may contain both necessary
process demand and a releasable safety margin.

The framework combines wastewater-process-informed state representation,
baseline reconstruction, input-reliability assessment, nitrogen-risk evaluation,
risk-constrained dosing adjustment, continuity control, and weekly posterior
feedback.

---

## Research concept

Historical operator actions contain valuable process knowledge, but they may also
reflect risk aversion, operational inertia, and conservative safety margins.
Directly learning and reproducing these actions can therefore preserve unnecessary
carbon dosing.

KBDC addresses this problem through the following logic:

```text
Historical operator dosing
        ↓
Conservative operational baseline
        ↓
Process-informed state representation
        ↓
Baseline reconstruction with machine learning
        ↓
Input-reliability assessment (VFC / QC)
        ↓
Nitrogen-risk evaluation
        ↓
Reduce / Retain / Protect decision
        ↓
Continuity correction
        ↓
Daily KBDC recommendation
        ↓
Weekly posterior feedback
        ↓
Next-week parameter adaptation
The aim is not to predict an assumed optimal dose directly. Instead, the learned
historical baseline acts as a safety anchor, from which conservative dosing margin
is released only when nitrogen risk and input reliability permit.

Study workflow
The code repository follows the main analytical workflow of the manuscript and
Supplementary Information:

Operational data
      │
      ▼
01  Leakage-aware preprocessing
      │
      ▼
02  Process-informed feature construction
      │
      ▼
03  Baseline model benchmarking and selection
      │
      ├──────────────► 04  SHAP attribution
      │
      ├──────────────► 05  Robustness analysis + VFC
      │
      └──────────────► 06  Quantile-GBR diagnostic
                              │
                              ▼
                    07  Sequential KBDC control
                              │
                              ▼
                    08  KNN historical reference
                         and field validation
The repository is organized as modular research scripts rather than as a single
unattended end-to-end program. This design reflects the actual study workflow,
especially the sequential control, field-QC, and weekly-feedback stages.

Repository structure
KBDC-WWTP/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── 01_preprocessing.py
├── 02_process_features.py
├── 03_model_benchmark.py
├── 04_shap_attribution.py
├── 05_robustness_vfc.py
├── 06_quantile_gbr.py
├── 07_kbdc_sequential_control.py
├── 08_knn_validation.py
│
└── data/
Code modules
01_preprocessing.py
Implements leakage-aware preprocessing of the historical WWTP dataset.

Main functions include:

chronological data partitioning;

training-only preprocessing;

variable-specific missing-value handling;

training-only outlier screening;

prevention of target leakage;

deployable-variable screening;

preparation of model-ready datasets.

Data-dependent preprocessing statistics are estimated from the training segment
and then fixed for later held-out and validation data.

Corresponding study components: Supplementary Text S1, Table S1, and Fig. S1.

02_process_features.py
Constructs the wastewater-process-informed state representation used for
baseline learning.

The engineered descriptors represent complementary aspects of the pre-dosing
process state, including:

hydraulic scale and low-flow pressure;

apparent carbon-nitrogen limitation;

delayed nitrogen-safety feedback;

influent nitrogen pressure relative to sludge-associated support.

The feature-construction layer uses information available at the dosing-decision
time and preserves the process meaning of the original operational variables.

Boundary-relative nitrogen descriptors are not clipped during PI feature
construction, so values above the plant limit retain the magnitude of the
exceedance.

Corresponding study components: Supplementary Text S2, Table S2, and Fig. S2.

03_model_benchmark.py
Benchmarks the baseline-reconstruction performance of multiple regression model
families using the predefined temporal validation protocol.

The comparison includes:

Linear Regression;

Decision Tree;

Support Vector Regression;

K-Nearest Neighbors;

Random Forest;

Gradient Boosting Regression;

Deep Neural Network;

Long Short-Term Memory network.

The analysis compares model performance under the predefined original and
process-informed representations and supports selection of the final
process-informed GBR model.

Corresponding study components: Supplementary Text S3, Table S3, and Fig. S3.

04_shap_attribution.py
Performs SHAP-based interpretation of the final process-informed GBR baseline
model.

The analysis provides:

global feature attribution;

feature-importance ranking;

sample-level attribution analysis;

interpretation of the contribution of process-informed descriptors.

SHAP values are used as model-attribution diagnostics rather than as evidence of
causal process relationships.

Corresponding study components: Supplementary Text S4 and Figs. S4-S5.

05_robustness_vfc.py
Evaluates model sensitivity to abnormal pre-dosing inputs and implements the
Virtual Flow Checking (VFC) reliability analysis.

The module includes:

grouped input perturbation;

raw-input zeroing tests;

global multivariable random bidirectional perturbation;

recalculation of dependent process-informed features after perturbation;

comparison of alternative VFC lag configurations;

isolated TREAT_FLOW anomaly experiments;

comparison of model performance with and without VFC;

flow-amplitude anomaly analysis;

VFC reliability-trigger assessment.

Global multivariable perturbation is used to characterize broader input
deterioration, whereas isolated treated-flow perturbation represents the intended
VFC application scenario.

VFC is used as a reliability safeguard rather than as an unconditional substitute
for measured treated flow.

Corresponding study components: Supplementary Text S6, Table S4 Panel A,
and Figs. S6-S9.

06_quantile_gbr.py
Implements Quantile Gradient Boosting Regression for historical-domain
consistency diagnostics.

Three conditional quantile models are constructed:

P20
P50
P80
The resulting P20-P80 envelope is used to assess whether a reconstructed
historical baseline remains consistent with the historical operator-response
domain.

The quantile models provide a diagnostic envelope and do not directly determine
the final KBDC dose.

Corresponding study component: Supplementary Text S5.

07_kbdc_sequential_control.py
Implements the sequential KBDC recommendation workflow.

The daily workflow integrates:

Pre-dosing operational state
        ↓
VFC / QC reliability assessment
        ↓
Confirmed operational input
        ↓
Process-informed descriptors
        ↓
GBR baseline reconstruction
        ↓
Historical-domain diagnostic
        ↓
Nitrogen-risk scoring
        ↓
Reduce / Retain / Protect branch
        ↓
Continuity correction
        ↓
Final KBDC recommendation
        ↓
Daily state recording
The control logic uses the reconstructed historical operator dose as a
conservative safety anchor.

Depending on nitrogen risk and input reliability, KBDC can:

Reduce the conservative margin under sufficiently safe conditions;

Retain the reconstructed baseline when additional reduction is not justified;

Protect the process through a bounded upward adjustment under elevated risk.

Input reliability is checked before dosing adjustment. When a treated-flow
anomaly is flagged, field QC or operator confirmation can retain, correct, or
leave the input unresolved.

Unresolved flow uncertainty restricts carbon reduction rather than independently
triggering protective addition.

The script also implements weekly posterior feedback. Effluent nitrogen outcomes
from a completed week are used only to update the control parameters for the
following week; they do not retrospectively alter recommendations that were
already issued.

Corresponding study components: Supplementary Texts S7-S8, Table S4 Panel B,
Table S5, Algorithm S1, and Algorithm S2.

08_knn_validation.py
Implements the KNN-matched historical reference used for prospective
field-validation analysis.

The historical reference procedure includes:

construction of the validation-excluded historical library;

preprocessing of matching variables using historical-library statistics;

weighted distance calculation;

K-nearest-neighbor matching;

K = 5 historical reference construction;

similarity calculation;

historical dosing reference;

matched historical nitrogen reference;

nearest-neighbor safety diagnostics.

The KNN reference provides an operationally comparable historical benchmark for
the prospective validation period rather than an optimization target.

Corresponding study components: Supplementary Text S9, Table S6,
Algorithm S3, Fig. S10, and the associated field-validation analyses in Text S10.

Environment
The public code was prepared using:

Python 3.13.7
Install the required Python packages with:

pip install -r requirements.txt
The package versions used for the released analyses are specified in
requirements.txt.

Quick start
Clone the repository:

git clone https://github.com/Liu-qing5/KBDC-WWTP.git
cd KBDC-WWTP
Install dependencies:

pip install -r requirements.txt
The available command-line options for each script can be inspected using:

python 01_preprocessing.py --help
python 02_process_features.py --help
python 03_model_benchmark.py --help
python 04_shap_attribution.py --help
python 05_robustness_vfc.py --help
python 06_quantile_gbr.py --help
python 07_kbdc_sequential_control.py --help
python 08_knn_validation.py --help
Recommended execution order
For retrospective model development and analysis:

python 01_preprocessing.py
python 02_process_features.py
python 03_model_benchmark.py
python 04_shap_attribution.py
python 05_robustness_vfc.py
python 06_quantile_gbr.py
The KBDC control and validation stages are handled separately because they
involve sequential operational states, QC information, and posterior feedback.

Sequential KBDC operation
07_kbdc_sequential_control.py supports the sequential recommendation workflow.

Daily / prospective recommendation
python 07_kbdc_sequential_control.py recommend --input <input_file>
This mode generates KBDC recommendations using only the information available at
the dosing-decision stage.

Weekly closure and feedback
python 07_kbdc_sequential_control.py close-week --week-file <completed_week_file>
This mode reads posterior weekly nitrogen outcomes and updates the control state
for the following week.

Retrospective sequential replay
python 07_kbdc_sequential_control.py replay --input <validation_file> --reset-state --week-size 7
This mode sequentially replays a validation trajectory while preserving the
daily decision order and weekly feedback boundary.

KNN historical-reference analysis
The historical-reference and field-validation analysis is implemented in:

python 08_knn_validation.py --help
The analysis requires the corresponding historical modelling library and
prospective validation data.

Important implementation principles
1. Decision-time information boundary
Baseline learning uses only information available at the dosing-decision time.

Same-day influent, hydraulic, and sludge-state information can describe the
pre-dosing state, while effluent nitrogen information enters the predictive and
control workflow only as delayed feedback where appropriate.

Same-day posterior effluent outcomes are reserved for validation and subsequent
feedback.

2. Chronological rather than random validation
The historical operational trajectory is treated as time-series process data.

Data partitioning is chronological, and preprocessing parameters derived from
training data are fixed before application to later data.

This prevents information from future observations from leaking into model
development.

3. Process-informed descriptors are reconstructed after perturbation
During robustness analysis, perturbing a raw operational input can alter multiple
dependent process-informed descriptors.

These dependent descriptors are therefore recalculated after perturbation rather
than being treated as independent columns.

4. PI feature construction and KBDC risk scoring are different layers
Boundary-relative nitrogen descriptors used by the process-informed baseline
model retain their original magnitude during feature construction.

For example, a value above the plant limit may remain greater than 1 in the
PI-GBR representation.

Clipping to the interval [0, 1] is applied only when the corresponding
information enters the bounded KBDC risk-scoring layer.

5. VFC is a reliability safeguard
Measured TREAT_FLOW remains the primary hydraulic input under normal
conditions.

VFC is activated as a reliability safeguard when the measured flow is missing,
outside the predefined reliability domain, or shows sufficiently large
disagreement with the virtual-flow estimate.

A VFC trigger does not by itself imply that the virtual estimate should
automatically replace the measured value. Field QC and operator confirmation are
part of the reliability-resolution process.

6. Quantile-GBR is diagnostic rather than prescriptive
The P20-P80 Quantile-GBR envelope is used to evaluate historical-domain
plausibility.

It does not directly calculate the amount of carbon-dose reduction or protective
addition.

Final KBDC adjustment is determined by the risk-constrained control layer.

7. Weekly feedback is forward-only
Posterior weekly effluent information is used to update the control state for the
next week.

The update does not retrospectively alter recommendations already generated in
the completed week.

Data
The study contains two temporally separated datasets:

a historical operational trajectory used for model development and historical
reference construction;

an isolated prospective field-validation period used for sequential KBDC
evaluation.

The historical record covers 652 consecutive daily observations, followed by a
28-day prospective validation period that was excluded from model fitting and
historical-reference construction.

Data files used for the public analyses should be placed under the data/
directory according to the input paths specified by the corresponding scripts.

Detailed variable definitions, units, decision-time availability, preprocessing
rules, and analytical roles are provided in the manuscript and Supplementary
Information.

Reproducibility
The repository is intended to provide the principal computational procedures
used in the study, including:

leakage-aware preprocessing;

process-informed state construction;

baseline model benchmarking;

SHAP attribution;

perturbation robustness analysis;

VFC reliability assessment;

Quantile-GBR diagnostics;

sequential KBDC control;

weekly feedback;

KNN historical-reference construction;

field-validation analysis.

Random seeds and model configurations are fixed in the corresponding scripts
where applicable.

Because field-control and validation stages involve sequential operational data,
reliability states, and posterior feedback, the code is provided as modular
analysis stages rather than as a single fully unattended reproduction script.

Relationship to the manuscript
The public code is organized to follow the scientific structure of the
Supplementary Information:

Code	Main purpose	Manuscript / SI component
01_preprocessing.py	Leakage-aware data preparation	Text S1
02_process_features.py	Process-informed state representation	Text S2
03_model_benchmark.py	Baseline learning and model comparison	Text S3
04_shap_attribution.py	Model attribution	Text S4
06_quantile_gbr.py	Historical-domain diagnostic	Text S5
05_robustness_vfc.py	Robustness and VFC	Text S6
07_kbdc_sequential_control.py	Risk-constrained dosing and weekly feedback	Texts S7-S8
08_knn_validation.py	Historical reference and validation	Texts S9-S10
Citation
If you use this repository, please cite the associated manuscript:

Knowledge-embedded learning of conservative human dosing baselines for
risk-constrained carbon optimization in wastewater nitrogen removal

The complete journal citation and DOI will be added after publication.

Contact
For questions regarding the scientific methodology, data interpretation, or
implementation of KBDC, please refer to the corresponding authors listed in the
associated manuscript.
