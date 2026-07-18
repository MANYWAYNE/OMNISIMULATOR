# Profiles Directory
Place JSON camera fingerprint profiles here.
Each profile should be a JSON file with statistical fingerprints
derived from real camera samples.

Format: profile_name.json
{
  "r_mean": [...],
  "g_mean": [...],
  "b_mean": [...],
  "noise_sigma": 2.5
}

Example profiles:
- Sony_A7IV_Natural.json
- iPhone_15_ProMax.json
