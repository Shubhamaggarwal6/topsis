import numpy as np  

def topsis_score(matrix, weights, impacts):
    normalized_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))

    weighted_matrix = normalized_matrix * weights

    ideal_solution = np.zeros(weighted_matrix.shape[1])
    negative_ideal_solution = np.zeros(weighted_matrix.shape[1])

    for i in range(weighted_matrix.shape[1]):
        if impacts[i] == '+':
            ideal_solution[i] = np.max(weighted_matrix[:, i])
            negative_ideal_solution[i] = np.min(weighted_matrix[:, i])
        else:  
            ideal_solution[i] = np.min(weighted_matrix[:, i])
            negative_ideal_solution[i] = np.max(weighted_matrix[:, i])

    distance_to_ideal = np.sqrt(((weighted_matrix - ideal_solution) ** 2).sum(axis=1))
    distance_to_negative_ideal = np.sqrt(((weighted_matrix - negative_ideal_solution) ** 2).sum(axis=1))

    scores = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)

    return scores

def main():
    decision_matrix = np.array([[250, 16, 12, 5],
                                [200, 16, 8, 3],
                                [300, 32, 16, 4],
                                [300, 32, 16, 4],
                                [275, 32, 8, 4],
                                [225, 16, 16, 2]])
    
    num_criteria = decision_matrix.shape[1]

    weights = np.zeros(num_criteria)
    for i in range(num_criteria):
        while True:
            try:
                weight = float(input(f"Enter weight for criterion {i + 1} (value should be <= 1): "))
                if weight <= 1:
                    weights[i] = weight
                    break
                else:
                    print("Weight must be 1 or less. Please try again.")
            except ValueError:
                print("Invalid input. Enter a numerical value.")

    impacts = []
    for i in range(num_criteria):
        while True:
            impact = input(f"Enter impact for criterion {i + 1} ('+' for positive, '-' for negative): ").strip()
            if impact in ['+', '-']:
                impacts.append(impact)
                break
            else:
                print("Invalid impact. Enter '+' or '-'.")

    performance_scores = topsis_score(decision_matrix, weights, impacts)
    rankings = np.argsort(performance_scores)[::-1] + 1  # Rankings from highest to lowest

    print("Performance Scores:", performance_scores)
    print("Rankings:", rankings)

if __name__ == '__main__':
    main()
