# ai-router: the static CGO_ENABLED=0 binary on distroless static (nonroot,
# uid 65532). HomeLab mounts /etc/ai-router/config.yaml, the OpenBao secrets
# files and the /var/lib/ai-router pin store; the binary reads its secrets
# files itself, so the image needs no shell. Bump the digest with each base roll.
FROM gcr.io/distroless/static-debian13:nonroot@sha256:e2e927ec666bae08560abb3c55d0659eceabb657f56b6782ab500a9fc7f555e3

COPY --chmod=755 dist/ai-router-linux-amd64 /usr/local/bin/ai-router
USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/ai-router"]
CMD ["serve", "--config", "/etc/ai-router/config.yaml"]
