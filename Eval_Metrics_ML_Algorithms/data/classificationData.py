# binary balanced
# labelNames = ['spam', 'ham']
# realLabels = ['spam', 'spam', 'ham', 'ham', 'spam', 'ham']
# computedLabels = ['spam', 'ham', 'ham', 'spam', 'spam', 'ham']


# binary unbalanced
# labelNames = ['normal', 'infected']
# realLabels = ['infected', 'infected', 'infected', 'infected', 'normal', 'normal', 'normal', 'normal', 'normal','normal', 'normal', 'normal', 'normal', 'normal', 'normal']
# computedLabels = ['infected', 'infected', 'normal', 'normal', 'normal', 'normal','normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'infected']

# multi class
# labelNames = ['apple', 'pear', 'peach']
# realLabels = ['peach', 'apple', 'pear', 'apple', 'pear', 'peach']
# computedLabels = ['apple', 'apple', 'pear', 'pear', 'pear', 'peach']

# probabilities binary
# labelNames = ['spam', 'ham']
# realLabels = ['spam', 'spam', 'ham', 'ham', 'spam', 'ham']
# computedLabels = ['spam', 'ham', 'ham', 'spam', 'spam', 'ham']
# computedOutputs = [[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1], [0.7, 0.3], [0.4, 0.6]]

# probabilities multi class
# labelNames = ['spam', 'ham', 'jam']
# realLabels = ['spam', 'jam', 'spam', 'jam', 'ham', 'ham']
# computedLabels = ['spam', 'ham', 'spam', 'jam', 'ham', 'jam']
# computedOutputs = [[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.5, 0.25, 0.25], [0.1, 0.1, 0.8], [0.2, 0.7, 0.1],
#                    [0.2, 0.2, 0.6]]

# probabilities multi label
labelNames = ['spam', 'ham', 'jam']
realLabels = [[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 0]]
computedLabels = []
computedOutputs = [[0.7, 0.6, 0.3], [0.5, 0.6, 0.2], [0.3, 0.9, 0.7], [0.7, 0.9, 0.4]]
