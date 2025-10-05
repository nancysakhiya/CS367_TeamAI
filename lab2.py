import heapq
import re
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize


# Preprocessing function: normalize text by lowercasing and removing punctuation
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    sentences = sent_tokenize(text)  # Tokenize into sentences
    return sentences

# Function to compute Levenshtein Distance (edit distance)
def levenshtein_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]

# Heuristic function: estimate remaining alignment cost (sum of minimum edit distances)
def heuristic(doc1_sentences, doc2_sentences, i, j):
    remaining_cost = 0
    for x, y in zip(range(i, len(doc1_sentences)), range(j, len(doc2_sentences))):
        remaining_cost += levenshtein_distance(doc1_sentences[x], doc2_sentences[y])
    return remaining_cost

# A* search function for sentence alignment
def a_star_search(doc1_sentences, doc2_sentences):
    start_state = (0, 0, 0)  # (index in doc1, index in doc2, accumulated cost)
    goal_state = (len(doc1_sentences), len(doc2_sentences))

    # Priority queue for A* search (min-heap)
    pq = []
    heapq.heappush(pq, (0, start_state))

    # Keep track of visited states to avoid reprocessing
    visited = set()

    while pq:
        current_f, (i, j, g) = heapq.heappop(pq)

        if (i, j) in visited:
            continue

        visited.add((i, j))

        # Check if goal is reached
        if (i, j) == goal_state:
            return g  # Return total alignment cost

        # Transitions: align sentences, skip a sentence from doc1 or doc2
        if i < len(doc1_sentences) and j < len(doc2_sentences):
            cost = levenshtein_distance(doc1_sentences[i], doc2_sentences[j])
            new_state = (i + 1, j + 1, g + cost)
            h = heuristic(doc1_sentences, doc2_sentences, i + 1, j + 1)
            heapq.heappush(pq, (new_state[2] + h, new_state))

        if i < len(doc1_sentences):
            new_state = (i + 1, j, g + 1)  # Skipping a sentence from doc1
            h = heuristic(doc1_sentences, doc2_sentences, i + 1, j)
            heapq.heappush(pq, (new_state[2] + h, new_state))

        if j < len(doc2_sentences):
            new_state = (i, j + 1, g + 1)  # Skipping a sentence from doc2
            h = heuristic(doc1_sentences, doc2_sentences, i, j + 1)
            heapq.heappush(pq, (new_state[2] + h, new_state))

    return float('inf')  # In case no valid alignment is found

# Plagiarism detection function
def detect_plagiarism(doc1, doc2):
    # Preprocess both documents
    doc1_sentences = preprocess(doc1)
    doc2_sentences = preprocess(doc2)

    # Perform A* search for alignment
    alignment_cost = a_star_search(doc1_sentences, doc2_sentences)

    # Heuristic threshold for detecting plagiarism (e.g., cost below a certain threshold)
    plagiarism_threshold = 10  # Customize based on experiment

    if alignment_cost <= plagiarism_threshold:
        print(f"Potential plagiarism detected with alignment cost: {alignment_cost}")
    else:
        print(f"No significant plagiarism detected. Alignment cost: {alignment_cost}")

# Test cases for the plagiarism detection system
def run_tests():
    # Test Case 1: Identical Documents
    doc1 = "This is a test. It is only a test."
    doc2 = "This is a test. It is only a test."
    detect_plagiarism(doc1, doc2)

    # Test Case 2: Slightly Modified Document
    doc1 = "This is a test. It is only a test."
    doc2 = "This is a test. This is just a test."
    detect_plagiarism(doc1, doc2)

    # Test Case 3: Completely Different Documents
    doc1 = "This is a test. It is only a test."
    doc2 = "Plagiarism detection is important."
    detect_plagiarism(doc1, doc2)

    # Test Case 4: Partial Overlap
    doc1 = "This is a test. It is only a test."
    doc2 = "This is a test. Plagiarism detection is important."
    detect_plagiarism(doc1, doc2)

# Run the test cases
run_tests()