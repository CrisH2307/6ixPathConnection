**Service**

```
python3 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

```
Edge weights = 
    W_SKILL          * skill_sim
  + W_SCHOOL         * same_school
  + W_COMPANY        * same_company
  + W_RELATION       * relationship_strength
  + W_COMPANY_AFF    * company_affinity
  + W_SENIORITY_GAP  * seniority_score
```
