from sca import SemanticConsistencyAnalyzer


results = SemanticConsistencyAnalyzer(
      model="claude-sonnet-4-6",
      prompt="What causes inflation?",
      n=20,
      temperature=0.9,
  ).run()


print(results.similarity_matrix)
