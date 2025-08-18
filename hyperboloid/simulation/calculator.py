import json
import sympy as sp
import numpy as np
import argparse
import sys
from tabulate import tabulate

def load_equations(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading equations: {e}")
        sys.exit(1)

def parse_input_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        variables = data.get('variables', [])
        for var in variables:
            if 'expression' in var:
                try:
                    var['value'] = sp.sympify(var['expression'], locals={'t': sp.Symbol('t'), 'log': sp.log})
                except Exception as e:
                    print(f"Invalid expression for {var['name']}: {e}")
                    var['value'] = None
        return variables
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return None

def parse_command_line_input():
    variables = []
    print("Enter variables (format: name=value unit [expression], e.g., I=0.5 A or I=0.5*t A expression). Type 'done' to finish:")
    while True:
        inp = input("> ")
        if inp.lower() == 'done':
            break
        try:
            parts = inp.strip().split()
            name = parts[0].split('=')[0]
            value_part = parts[0].split('=')[1]
            unit = parts[1]
            expression = parts[2] if len(parts) > 2 and parts[2] == 'expression' else None
            value = sp.sympify(value_part, locals={'t': sp.Symbol('t'), 'log': sp.log}) if expression else float(value_part)
            variables.append({'name': name, 'value': value, 'unit': unit})
        except Exception as e:
            print(f"Invalid input: {e}. Try again.")
    return variables

def validate_units(variables, equation):
    required_units = {var: info['unit'] for var, info in equation['variables'].items()}
    input_units = {var['name']: var['unit'] for var in variables}
    missing = []
    mismatches = []
    for var in required_units:
        if var not in input_units and 'value' not in equation['variables'][var]:
            missing.append(f"{var} (unit: {required_units[var]})")
        elif var in input_units and input_units[var] != required_units[var]:
            mismatches.append(f"{var}: expected {required_units[var]}, got {input_units[var]}")
    if missing or mismatches:
        return False, f"Missing: {', '.join(missing)}\nMismatches: {', '.join(mismatches)}"
    return True, ""

def compute_equation(equation, variables):
    try:
        # Define symbols
        symbols = {var: sp.Symbol(var) for var in equation['variables']}
        expr = sp.sympify(equation['formula'], locals={**symbols, 'mean': np.mean, 'abs': abs, 'exp': sp.exp, 'log': sp.log})
        
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
        if isinstance(result, sp.Expr) and result.free_symbols:
            return None, f"Cannot compute: unresolved symbols {result.free_symbols}"
        result = float(result) if result.is_number else result
        
        # Compute derivatives if time-dependent
        derivation = [f"Formula: {equation['formula']}"]
        derivation.append("Substituted values:")
        for var, val in subs_dict.items():
            derivation.append(f"  {var} = {val} {equation['variables'].get(str(var), {}).get('unit', '')}")
        if isinstance(result, sp.Expr) and sp.Symbol('t') in result.free_symbols:
            deriv = sp.diff(result, sp.Symbol('t'))
            derivation.append(f"Derivative (d/dt): {deriv}")
            if deriv.is_number:
                derivation.append(f"Evaluated derivative: {float(deriv)}")
        derivation.append(f"Result: {result} {list(equation['variables'].values())[0]['unit']}")
        
        return result, '\n'.join(derivation)
    except Exception as e:
        return None, f"Calculation error: {e}"

def main():
    parser = argparse.ArgumentParser(description="Physics Calculator for Toroidal Ferrocella")
    parser.add_argument('--json', type=str, help='Path to input JSON file')
    parser.add_argument('--mode', choices=['standard', 'aether'], default='standard', help='Equation mode')
    args = parser.parse_args()

    # Load equations
    eq_file = f'simulation/{args.mode}_equations.json'
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
    print(tabulate(
        [[v['name'], v['value'], v['unit']] for v in variables],
        headers=['Name', 'Value', 'Unit'],
        tablefmt='grid'
    ))

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
        result, derivation = compute_equation(eq, variables)
        if derivation:
            print(f"Derivation:\n{derivation}")
        else:
            print(f"Error: {error}")

if __name__ == "__main__":
    main()
