import json
import sympy as sp
import numpy as np
import argparse
import sys
from tabulate import tabulate

def load_equations(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def parse_input_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data['variables']
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return None

def parse_command_line_input():
    variables = []
    print("Enter variables (format: name=value unit, e.g., I=0.5 A). Type 'done' to finish:")
    while True:
        inp = input("> ")
        if inp.lower() == 'done':
            break
        try:
            name, rest = inp.split('=', 1)
            value, unit = rest.strip().split(' ', 1)
            variables.append({'name': name.strip(), 'value': float(value), 'unit': unit.strip()})
        except Exception as e:
            print(f"Invalid input: {e}. Try again.")
    return variables

def validate_units(variables, equation):
    required_units = {var: info['unit'] for var, info in equation['variables'].items()}
    input_units = {var['name']: var['unit'] for var in variables}
    for var in required_units:
        if var not in input_units and 'value' not in equation['variables'][var]:
            return False, f"Missing variable: {var} (unit: {required_units[var]})"
        if var in input_units and input_units[var] != required_units[var]:
            return False, f"Unit mismatch for {var}: expected {required_units[var]}, got {input_units[var]}"
    return True, ""

def compute_equation(equation, variables):
    try:
        # Define symbols
        symbols = {var: sp.Symbol(var) for var in equation['variables']}
        expr = sp.sympify(equation['formula'], locals=symbols)
        
        # Substitute known values
        subs_dict = {}
        for var in equation['variables']:
            if 'value' in equation['variables'][var]:
                subs_dict[symbols[var]] = equation['variables'][var]['value']
            for v in variables:
                if v['name'] == var:
                    subs_dict[symbols[var]] = v['value']
        
        # Check for missing variables
        missing = [var for var in symbols if symbols[var] not in subs_dict]
        if missing:
            return None, f"Missing values for: {', '.join(missing)}"
        
        # Compute result
        result = expr.subs(subs_dict)
        if result.is_number:
            result = float(result)
        else:
            return None, "Result is not a number (complex or symbolic)"
        
        # Show derivation
        print("\nDerivation:")
        print(f"Formula: {equation['formula']}")
        print("Substituted values:")
        for var, val in subs_dict.items():
            print(f"  {var} = {val} {equation['variables'].get(str(var), {}).get('unit', '')}")
        print(f"Result: {result} {list(equation['variables'].values())[0]['unit']}")
        
        return result, None
    except Exception as e:
        return None, f"Calculation error: {e}"

def main():
    parser = argparse.ArgumentParser(description="Physics Calculator for Toroidal Ferrocella")
    parser.add_argument('--json', type=str, help='Path to input JSON file')
    parser.add_argument('--mode', choices=['standard', 'aether'], default='standard', help='Equation mode')
    args = parser.parse_args()

    # Load equations
    eq_file = 'simulation/standard_equations.json' if args.mode == 'standard' else 'simulation/aether_equations.json'
    equations = load_equations(eq_file)

    # Get variables
    if args.json:
        variables = parse_input_json(args.json)
        if not variables:
            sys.exit(1)
    else:
        variables = parse_command_line_input()
        if not variables:
            print("No variables provided.")
            sys.exit(1)

    # Display input variables
    print("\nInput Variables:")
    print(tabulate(variables, headers=['name', 'value', 'unit'], tablefmt='grid'))

    # Select equations
    print("\nAvailable Equations:")
    for i, eq in enumerate(equations, 1):
        print(f"{i}. {eq['name']}: {eq['formula']}")
    eq_indices = input("Enter equation numbers (comma-separated, e.g., 1,2,3): ").split(',')
    try:
        eq_indices = [int(i) - 1 for i in eq_indices]
    except ValueError:
        print("Invalid equation numbers.")
        sys.exit(1)

    # Compute results
    print("\nResults:")
    for idx in eq_indices:
        if idx < 0 or idx >= len(equations):
            print(f"Invalid equation index: {idx + 1}")
            continue
        eq = equations[idx]
        print(f"\nEquation: {eq['name']}")
        valid, error = validate_units(variables, eq)
        if not valid:
            print(f"Error: {error}")
            continue
        result, error = compute_equation(eq, variables)
        if error:
            print(f"Error: {error}")
        else:
            print(f"Result: {result} {list(eq['variables'].values())[0]['unit']}")

if __name__ == "__main__":
    main()
