Use uv for environment/dependency management.

For code style, prefer wider rows (up to 120 chars)

# Data model

- Service -- website, app, etc.
- Case -- A fact about a privacy policy that a user might want to know. For example, "No third-party analytics or tracking platforms are used"
    - Foreign key topic
- Point -- a plain text highlight that summarizes as aspect of a Service's privacy policy.
    - It is often a quote, but doesn't have to be, e.g. `The terms for this service are easy to read`
    - Many-to-1 with Cases
    - Foreign keys user, service, case, document
- Topic -- semantic high-level groupings of Cases. For example, "User Choice"
- Document -- crawled privacy policy or ToS from a Service
    - Foreign keys user, service
