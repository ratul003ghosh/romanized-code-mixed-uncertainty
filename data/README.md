# Data Policy

1. **Existing datasets**: Stored in `raw/` and `processed/`.
2. **Processed data**: Normalized and formatted datasets.
3. **Synthetic examples**: Used for local testing and PII validation without risk.
4. **Teacher-generated silver data**: Outputs from the teacher ensemble used to train the student.
5. **Gold evaluation data**: Manually verified evaluation sets.

**CRITICAL POLICY:**
- DO NOT commit real PII.
- DO NOT commit API keys, passwords, credentials, or `.env` files.
- DO NOT commit large model checkpoints.
- DO NOT commit private datasets that cannot legally be shared.
