# Whisper Transcription Helm Chart

## To use Helm Chart from Rancher:

1. Under `Apps`>`Repositories` create a new repository with the URL: https://southernmethodistuniversity.github.io/whisper-transcription/
2. Under `Apps`>`Charts` click on the application `whisper-transcription`
3. Select an appropriate `Namespace`, give the app a unique `Name` and check the box next to `Customize Helm options before install`
4. Update the `cname`, `frontend.image.tag`, `backend.image.tag`, and the `splunk.secret` and deploy

## To use Helm Chart from CLI

1. Add and update the Helm Repository:

```bash
helm repo add whisper-transcription https://southernmethodistuniversity.github.io/whisper-transcription/
helm repo update
```

2. Install the chart with your custom values:

```bash
helm install <release-name> whisper-transcription/whisper-transcription \
  --namespace <namespace> \
  --create-namespace \
  --set cname="your.custom.domain" \
  --set frontend.image.tag="your-frontend-tag" \
  --set backend.image.tag="your-backend-tag" \
  --set splunk.secret="your-splunk-secret"
```
