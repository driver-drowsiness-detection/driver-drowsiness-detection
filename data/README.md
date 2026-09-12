# Data

Arushuuu owns the dataset process. Vishruth reviews compatibility with landmarks, eye crops and fair EAR/CNN comparison. Both seniors decide labels, partitions and dataset suitability.

1. raw: original authorised files, ignored by Git.
2. processed: derived crops and features, ignored by Git.
3. demo: consenting stationary lab recordings, ignored by Git.
4. manifests: versioned metadata and split definitions. Review paths and identifying information before committing.

Before downloading a large dataset, record its source URL, licence, labels, subject identifiers, format, size and suitability. Eye crops alone might support CNN training but cannot necessarily support the EAR comparison.

Keep each subject in one partition. Both eyes and adjacent frames inherit their source subject's split. Training fits weights, calibration maps scores, validation chooses settings, and final test measures the frozen result.

No dataset has been selected or validated. Do not invent sample labels or split counts.
