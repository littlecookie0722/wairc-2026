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

An empty `captures` array is normalized to the format's implicit capture at
sample zero. Metadata labels remain strings; they are not converted into the
competition's nine-class `0..8` mapping.

The current parser intentionally rejects real-only datatypes, multiple
channels, non-zero `core:offset`, trailing bytes, capture headers, non-empty
extensions, unsupported specification major versions, and dataset paths that
leave the metadata directory. It does not load `.sigmf-data` bytes yet and it
does not interpret Collection files, extension namespaces, geolocation,
frequency ranges, or annotation semantics beyond the fields listed above.

The checked-in
[`minimal.sigmf-meta`](../tests/fixtures/sigmf/minimal.sigmf-meta) is a
synthetic metadata-only fixture. The next interoperability phase can add a
read-only raw-data adapter against this contract without changing the
competition or synthetic adapters.
