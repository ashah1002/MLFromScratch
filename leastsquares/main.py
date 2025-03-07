import numpy as np

def generate_noisy_polynomial_data(n_terms, num_points=10, x_range=(1, 10), coeff_range=(-10, 10)):
    coefficients = np.random.uniform(coeff_range[0], coeff_range[1], n_terms +1)
    
    terms = []
    for i, coef in enumerate(coefficients):
        power = n_terms - i
        if power > 1:
            terms.append(f"{coef:.2f}x^{power}")
        elif power == 1:
            terms.append(f"{coef:.2f}x")
        else:
            terms.append(f"{coef:.2f}")
    
    polynomial_str = " + ".join(terms).replace("+-", "- ")
    print(f"Generated Polynomial Function: f(x) = {polynomial_str}")

    # Generate noisy data points
    points = []
    for _ in range(num_points):
        x = np.random.uniform(*x_range)  # Random x in the given range
        y = sum(coefficients[i] * x**(n_terms - i) for i in range(n_terms + 1)) # y = f(x)
        noise = np.random.uniform(-2, 2)  # Random noise
        y += noise
        points.append((x, y))

    return points

def least_squares(n_terms, points):
    A = np.array([[x**i for i in range(n_terms, -1, -1)] for x, y in points])
    B = np.array([y for _, y in points])
    
    # Compute A^T * A and A^T * B
    A_T = A.T  # Transpose of A
    A_T_A = np.matmul(A_T, A)  # (A^T * A)
    A_T_B = np.matmul(A_T, B)  # (A^T * B)

    # Compute the inverse of (A^T * A)
    A_T_A_inv = np.linalg.inv(A_T_A)  # (A^T * A)^-1

    # Compute the least squares solution X = (A^T A)^-1 A^T B
    fitted_coefficients = np.matmul(A_T_A_inv, A_T_B)
    
    return fitted_coefficients

n_terms = 30
n_points = 100

points = generate_noisy_polynomial_data(n_terms, num_points=n_points)
fitted_coefficients = least_squares(n_terms, points)

print(f"Fitted Coefficients (highest degree first): {fitted_coefficients}")