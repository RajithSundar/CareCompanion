# Running CareCompanion on Azure AI

The coach's language model runs on **Azure OpenAI** (part of Azure AI Foundry). The app itself can run anywhere; the second half of this guide hosts it on Azure Container Apps so the whole demo is on Azure.

## 1. Create the model deployment

1. In the Azure portal, create an **Azure OpenAI** resource (not a Foundry project). The demo uses South India; student subscriptions only allow some regions.
2. Deploy `gpt-4.1-mini` (version 2025-04-14, Global Standard). `gpt-4o-mini` can no longer be deployed. If the Foundry portal keeps sending you to "create a project", deploy from Azure Cloud Shell instead, using the resource ID from the resource's JSON view:

   ```bash
   az rest --method put --url "https://management.azure.com<resource-id>/deployments/gpt-4.1-mini?api-version=2024-10-01" \
     --body '{"sku":{"name":"GlobalStandard","capacity":100},"properties":{"model":{"format":"OpenAI","name":"gpt-4.1-mini","version":"2025-04-14"}}}'
   ```

   Capacity is in thousands of tokens per minute (100 = 100 requests and 100k tokens a minute). One coach reply makes about 5 requests, so 10 is too low.
3. In the portal, open the resource, then **Keys and Endpoint**. Copy the endpoint (`https://<name>.openai.azure.com/`) and one key.

## 2. Point the app at it

In `.env`:

```bash
AZURE_OPENAI_ENDPOINT=https://<name>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_DEPLOYMENT=<deployment name>
AZURE_OPENAI_API_VERSION=2024-10-21
```

Restart the app. The header button reads **Coach: Azure OpenAI**, and every coach reply is tagged with the deployment that wrote it. Azure OpenAI takes priority over the other providers when all three variables are set.

## 3. Host the app on Azure Container Apps (optional)

With the Azure CLI signed in (`az login`) and the `containerapp` extension installed:

```bash
az group create -n carecompanion-rg -l <region>
az containerapp up \
  --name carecompanion \
  --resource-group carecompanion-rg \
  --source . \
  --ingress external --target-port 8000 \
  --env-vars AZURE_OPENAI_ENDPOINT=<endpoint> AZURE_OPENAI_DEPLOYMENT=<deployment> \
             AZURE_OPENAI_API_VERSION=2024-10-21 AZURE_OPENAI_API_KEY=secretref:aoai-key
az containerapp secret set -n carecompanion -g carecompanion-rg --secrets aoai-key=<key>
```

`az containerapp up` builds the `Dockerfile` in Azure and prints the public URL.

Notes:
- Visitor sandboxes live in the container's filesystem, so they reset when the container restarts. That is fine for a demo; mount an Azure Files volume at `/app/data/sandboxes` to keep them.
- Rate limits are per process and keyed on the client IP from `X-Forwarded-For`. Behind Container Apps ingress the header is trusted from any proxy, so treat the limits as a cost guard, not a security boundary; set a spending limit on the Azure OpenAI resource as well.
- The image includes the trained model and the derived PMData daily table (`data/pmdata_daily.csv`, CC BY 4.0, credited in the footer and README), not the raw dataset.

## 4. The deliverable: advice changing across sessions

Open `/demo`, pick a profile (a fictional persona or one of the 16 real PMData participants), and run the three sessions. Each session shows that day's readings, what the person said, the coach's reply (Azure OpenAI, with memory and earlier sessions), the reply a no-memory assistant gives to the same question, and the facts the coach remembered going in and saved coming out.
