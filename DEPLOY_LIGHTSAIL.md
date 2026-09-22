# Deploying to AWS Lightsail Containers (Frankfurt, eu-central-1)

Run every command below in your own Mac's **Terminal app**, not through Claude, since
this needs real internet access to AWS and your AWS secret key should never be pasted
into a chat. Each step tells you what to expect.

## 1. Install Docker Desktop

Download from https://www.docker.com/products/docker-desktop/, install it, open it once
so it finishes starting up. Confirm it's working:

```bash
docker --version
```

## 2. Install the AWS CLI

```bash
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
aws --version
```

## 3. Install the Lightsail CLI plugin (`lightsailctl`)

This is a separate small plugin the AWS CLI needs specifically for pushing container
images to Lightsail.

```bash
curl "https://s3.us-west-2.amazonaws.com/lightsailctl/latest/darwin-arm64/lightsailctl" -o "lightsailctl"
chmod +x lightsailctl
sudo mv lightsailctl /usr/local/bin/lightsailctl
```

(If your Mac is Intel, not Apple Silicon, use `darwin-amd64` instead of `darwin-arm64`
in the URL above.)

## 4. Configure the AWS CLI with your IAM user's access key

You should already have the **Access key ID** and **Secret access key** from creating
the `lightsail-deploy` IAM user. Run:

```bash
aws configure
```

It will prompt you for four things:
- **AWS Access Key ID**: paste it
- **AWS Secret Access Key**: paste it
- **Default region name**: `eu-central-1`
- **Default output format**: `json`

Verify it worked (this should print your account info, no secrets shown):

```bash
aws sts get-caller-identity
```

## 5. Build the Docker image

From inside your project folder (`EPF_Dashboard-Expert-Review`):

```bash
cd "/Users/irinalazar/Documents/ELECTA KU LEUVEN/EPF_Dashboard-Expert-Review"
docker build -t epf-review .
```

This uses the `Dockerfile` already in that folder. Takes a minute or two the first time.

## 6. Create the Lightsail container service

```bash
aws lightsail create-container-service \
  --service-name epf-review \
  --power micro \
  --scale 1 \
  --region eu-central-1
```

`--power micro` is the $10/mo tier (1GB RAM) we settled on. This takes a minute to
provision, you can check on it with:

```bash
aws lightsail get-container-services --service-name epf-review --region eu-central-1
```

Wait until `"state"` shows `"READY"` before continuing.

## 7. Push your image to Lightsail

```bash
aws lightsail push-container-image \
  --service-name epf-review \
  --label epf-review \
  --image epf-review:latest \
  --region eu-central-1
```

The output ends with a line like:
```
Refer to this image as ":epf-review.epf-review.1" in deployments.
```
**Copy that exact string** (yours may end in a different number), you need it in the
next step.

## 8. Create the deployment (this is where your secrets go)

Create a file called `deployment.json` in the same folder:

```bash
cat > deployment.json << 'EOF'
{
  "serviceName": "epf-review",
  "containers": {
    "epf-review": {
      "image": "PASTE_YOUR_IMAGE_REFERENCE_HERE",
      "ports": { "8080": "HTTP" },
      "environment": {
        "DATABASE_URL": "PASTE_YOUR_SUPABASE_CONNECTION_STRING_HERE",
        "GITHUB_TOKEN": "PASTE_YOUR_GITHUB_TOKEN_HERE"
      }
    }
  },
  "publicEndpoint": {
    "containerName": "epf-review",
    "containerPort": 8080,
    "healthCheck": {
      "path": "/_stcore/health",
      "intervalSeconds": 30,
      "timeoutSeconds": 5,
      "healthyThreshold": 2,
      "unhealthyThreshold": 3,
      "successCodes": "200"
    }
  }
}
EOF
```

Then **edit `deployment.json`** (open it in any text editor, or `nano deployment.json`)
and replace the three `PASTE_...` placeholders:
- The image reference from step 7's output
- Your Supabase `DATABASE_URL` (Supabase dashboard -> Project Settings -> Database ->
  Connection string -> URI, the same one from your `.env` file)
- Your `GITHUB_TOKEN` (also from your `.env` file)

Then deploy it:

```bash
aws lightsail create-container-service-deployment \
  --region eu-central-1 \
  --cli-input-json file://deployment.json
```

This takes a few minutes. Check progress with the same command from step 6:

```bash
aws lightsail get-container-services --service-name epf-review --region eu-central-1
```

Look for `"state": "RUNNING"` and a `"url"` field, that's your live app.

## 9. Open it

```bash
aws lightsail get-container-services --service-name epf-review --region eu-central-1 \
  --query 'containerServices[0].url' --output text
```

Open that URL in your browser and log in.

## Day-to-day after this

**Redeploying after a code change**: repeat steps 5, 7, and 8 (build -> push -> deploy).
The service itself (step 6) only needs to be created once.

**Checking logs** if something looks wrong:
```bash
aws lightsail get-container-log \
  --service-name epf-review \
  --container-name epf-review \
  --region eu-central-1
```

**Deleting the service** (if you ever want to stop paying for it):
```bash
aws lightsail delete-container-service --service-name epf-review --region eu-central-1
```

**Keep `deployment.json` out of git**: it has your real secrets in it once you fill it
in. Add it to `.gitignore` before committing anything:
```bash
echo "deployment.json" >> .gitignore
```

## What's already been verified (from Claude's side)

The `Dockerfile` itself, and the exact command it runs, was tested end-to-end in a
clean environment against a real Postgres database (dependency install, app boot,
health check all passed), and separately load-tested under CPU/memory constraints
matching Lightsail's smallest tiers (see the earlier conversation for those numbers,
everything stayed under ~2 seconds even throttled to 0.25 vCPU). What hasn't been
tested is the actual `docker build` / `aws lightsail` command sequence above, since this
environment's network can't reach Docker Hub or AWS's API, that part runs for the first
time when you follow the steps above in your own Terminal.
