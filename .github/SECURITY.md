# Security Policy

## Supported versions

Security fixes are applied to the latest release only.

| Version | Supported |
|---------|-----------|
| 1.13.x  | ✅        |
| < 1.13  | ❌        |

## Reporting a vulnerability

Please do not report security vulnerabilities in a public GitHub Issue. Instead, use a
[private security advisory](https://github.com/itcojo/contrail-detect/security/advisories/new),
which keeps the details private until a fix is available. If you cannot use the advisory form,
contact the maintainers at jonathan.itcovitz@contrails.org.

We aim to acknowledge reports within 5 business days, and to provide a fix or mitigation
within 30 days where possible, depending on severity and complexity.

## Scope

The package processes satellite imagery and flight data locally. The following are in scope:

- the handling of credentials (Contrails.org, Google, and CDS API keys), including anything
  that could cause them to be leaked;
- known vulnerabilities in the package's dependencies;
- code paths that read untrusted files (e.g., user-supplied NetCDF, Parquet, or CSV files).

Issues with third-party services (the Contrails.org API, the Google Contrails API, and the
Copernicus Climate Data Store) should be reported to those providers directly.
