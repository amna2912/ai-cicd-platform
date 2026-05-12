import os
import httpx
from data.models import PipelineRun

async def post_prediction_to_gitlab(run: PipelineRun, prediction: dict):
    token = os.getenv("GITLAB_TOKEN")
    if not token:
        print("⚠️ GITLAB_TOKEN non défini")
        return

    project_id = run.repo_name.replace("/", "%2F")  

    comment = f"""## 🚀 **PipelineIQ AI Prediction**

**Risk Level**: **{prediction['risk_level']}** ({prediction['failure_probability']:.1%} de risque d'échec)

**Facteurs principaux :**
"""

    for factor in prediction.get("explanation", {}).get("top_factors", [])[:3]:
        direction = "🔴" if factor.get("shap", 0) > 0 else "🟢"
        comment += f"- {direction} `{factor['feature']}` = {factor['value']} (impact: {factor['shap']:+.3f})\n"

    comment += f"\n💡 **Recommandation** : Vérifier les tests et dépendances avant de merger."

    url = f"https://gitlab.com/api/v4/projects/{project_id}/pipelines/{run.external_id}/notes"

    headers = {"PRIVATE-TOKEN": token}
    payload = {"body": comment}

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code in (200, 201):
                print(f"✅ Commentaire posté sur GitLab pour pipeline {run.id}")
            else:
                print(f"❌ Erreur GitLab: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"❌ Erreur connexion: {e}")