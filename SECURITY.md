# Security

## Reporting a vulnerability

Please report security issues privately through GitHub's security advisory feature. Do not open a public issue containing credentials, private prompts, or exploitable request payloads.

## Deployment notes

- Set `GATEWAY_API_KEYS` in any shared environment. An empty value intentionally disables authentication.
- Put the gateway behind TLS, a private network, or an authenticated reverse proxy.
- Restrict `GATEWAY_CORS_ORIGINS` and `GATEWAY_TRUSTED_HOSTS` outside local development.
- The public-URL document loader rejects loopback, private, link-local, reserved, and internal-name targets to reduce SSRF risk.
- Treat prompts, tool results, uploaded documents, and model output as sensitive data. Configure log retention accordingly.
- Tool execution belongs in the calling agent. This gateway never executes model-requested tools itself.

