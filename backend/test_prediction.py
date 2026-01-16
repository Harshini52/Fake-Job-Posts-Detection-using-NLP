from job_predict import predict_job_post

# Fake job example
fake_text = """
Work from home job.
No interview required.
Drop your CV to gmail.com.
Earn 25,000 per month.
"""

# Real job example
real_text = """
We are hiring a Software Engineer.
Candidates must have 2+ years of experience.
Interview will be conducted in 2 rounds.
Apply through company career portal.
"""

print("Fake Job Test:", predict_job_post(fake_text))
print("Real Job Test:", predict_job_post(real_text))
