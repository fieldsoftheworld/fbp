"""Print the default variant names, one per line, for the job scripts.

Scoring every ``v:`` tag found on disk would also re-score variants written by
other jobs, so the scoring step is given this explicit list.
"""

from .variants import default_variants

if __name__ == "__main__":
    print(" ".join(default_variants()))
