# SigMF interoperability

The repository has an experimental parser for a deliberately small SigMF
metadata subset. It is based on the [SigMF core specification](https://github.com/sigmf/SigMF)
and does not redistribute a recording or claim complete SigMF compatibility.

## Current phase

`wairc_rf.sigmf.parse_sigmf_metadata` currently validates and returns:

- the required `global.core:datatype` and `global.core:version` fields;
- positive `global.core:sample_rate`;
- single-channel complex datatypes such as `ci16_le` and `cf32_le`;
- a normalized `captures` list containing `core:sample_start` and optional
  `core:frequency`;
- an `annotations` list containing `core:sample_start`, optional
  `core:sample_count`, and optional `core:label`;
- a same-directory `global.core:dataset` filename when one is declared.

`wairc_rf.sigmf.SigMFDatasetAdapter` uses that metadata to read one recording
as one unlabeled `RFSample` with one `RFNode`. It loads the data file only when
the sample is indexed, supports the complex integer interleaved and complex
float datatypes accepted by the parser, and preserves the metadata sample rate.

An empty `captures` array is normalized to the format's implicit capture at
sample zero. Metadata labels remain strings; they are not converted into the
competition's nine-class `0..8` mapping.

The current parser and adapter intentionally reject real-only datatypes, multiple
channels, non-zero `core:offset`, trailing bytes, capture headers, non-empty
extensions, unsupported specification major versions, and dataset paths that
leave the metadata directory. They do not interpret Collection files, extension namespaces, geolocation,
frequency ranges, or annotation semantics beyond the fields listed above.

The checked-in
[`minimal.sigmf-meta`](../tests/fixtures/sigmf/minimal.sigmf-meta) is a
synthetic metadata fixture. The adapter is deliberately single-record and
unlabeled; it does not change the competition or synthetic adapters, and it
does not map SigMF annotation strings to the competition label mapping.
