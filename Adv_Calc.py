import math
import os
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
import unittest

# Initialize global objects
console = Console()
session = PromptSession(">>> ", multiline=False)
menu_session = PromptSession("> ", multiline=False)
bindings = KeyBindings()
history = []  # List of (expression, result) tuples
CALCULATOR_DIR = "calculators"  # Directory for history files

# ---- Lexical Analysis ----
class Token:
    def __init__(self, type, value):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value})"

def tokenize(input_str):
    """Convert input string into a list of tokens."""
    tokens = []
    i = 0
    while i < len(input_str):
        char = input_str[i]
        if char.isspace():
            i += 1
            continue
        elif char.isdigit() or char == '.':
            num = ''
            while i < len(input_str) and (input_str[i].isdigit() or input_str[i] == '.'):
                num += input_str[i]
                i += 1
            try:
                tokens.append(Token('NUMBER', float(num)))
            except ValueError:
                raise ValueError(f"Invalid number: {num}")
            continue
        elif char.isalpha():
            ident = ''
            while i < len(input_str) and input_str[i].isalnum():
                ident += input_str[i]
                i += 1
            if ident in ['sin', 'cos', 'tan', 'sqrt', 'log', 'ln', 'pi', 'e', 'if']:
                tokens.append(Token('KEYWORD', ident))
            else:
                tokens.append(Token('IDENTIFIER', ident))
            continue
        elif char in '+-*/^=><':
            if char == '=' and i + 1 < len(input_str) and input_str[i + 1] == '=':
                tokens.append(Token('OPERATOR', '=='))
                i += 2
            else:
                tokens.append(Token('OPERATOR' if char != '=' else 'ASSIGNMENT', char))
                i += 1
        elif char in '(){':
            tokens.append(Token('PAREN' if char in '()' else 'BRACE', char))
            i += 1
        else:
            raise ValueError(f"Invalid character: {char}")
    tokens.append(Token('EOF', ''))
    return tokens

# ---- Syntax Analysis ----
class NumberNode:
    def __init__(self, value):
        self.value = value

class VarNode:
    def __init__(self, name):
        self.name = name

class BinaryOpNode:
    def __init__(self, left, op, right):
        self.left = left
        self.op = op
        self.right = right

class AssignNode:
    def __init__(self, var, value):
        self.var = var
        self.value = value

class FunctionNode:
    def __init__(self, func, arg):
        self.func = func
        self.arg = arg

class IfNode:
    def __init__(self, condition, body):
        self.condition = condition
        self.body = body

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def consume(self, token_type):
        if self.pos < len(self.tokens) and self.tokens[self.pos].type == token_type:
            self.pos += 1
            return self.tokens[self.pos - 1]
        raise ValueError(f"Expected {token_type}, got {self.tokens[self.pos].type if self.pos < len(self.tokens) else 'EOF'}")

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token('EOF', '')

    def factor(self):
        token = self.peek()
        if token.type == 'NUMBER':
            self.consume('NUMBER')
            return NumberNode(token.value)
        elif token.type == 'IDENTIFIER':
            self.consume('IDENTIFIER')
            return VarNode(token.value)
        elif token.type == 'KEYWORD' and token.value in ['sin', 'cos', 'tan', 'sqrt', 'log', 'ln']:
            self.consume('KEYWORD')
            self.consume('PAREN')
            arg = self.expression()
            self.consume('PAREN')
            return FunctionNode(token.value, arg)
        elif token.type == 'KEYWORD' and token.value in ['pi', 'e']:
            self.consume('KEYWORD')
            return NumberNode(math.pi if token.value == 'pi' else math.e)
        elif token.type == 'PAREN':
            self.consume('PAREN')
            expr = self.expression()
            self.consume('PAREN')
            return expr
        raise ValueError(f"Invalid factor: {token}")

    def term(self):
        node = self.factor()
        while self.peek().type == 'OPERATOR' and self.peek().value in ['*', '/', '^']:
            op = self.consume('OPERATOR').value
            right = self.factor()
            node = BinaryOpNode(node, op, right)
        return node

    def expression(self):
        node = self.term()
        while self.peek().type == 'OPERATOR' and self.peek().value in ['+', '-', '>', '<', '==']:
            op = self.consume('OPERATOR').value
            right = self.term()
            node = BinaryOpNode(node, op, right)
        return node

    def statement(self):
        if self.peek().type == 'KEYWORD' and self.peek().value == 'if':
            self.consume('KEYWORD')
            self.consume('PAREN')
            condition = self.expression()
            self.consume('PAREN')
            self.consume('BRACE')
            body = self.statement()
            self.consume('BRACE')
            return IfNode(condition, body)
        elif self.peek().type == 'IDENTIFIER' and self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == 'ASSIGNMENT':
            var = self.consume('IDENTIFIER').value
            self.consume('ASSIGNMENT')
            expr = self.expression()
            return AssignNode(var, expr)
        return self.expression()

# ---- Semantic Analysis ----
class SymbolTable:
    def __init__(self):
        self.symbols = {}

    def declare(self, var, value):
        self.symbols[var] = {'value': value, 'type': type(value).__name__}

    def get(self, var):
        if var not in self.symbols:
            raise ValueError(f"Undeclared variable: {var}")
        return self.symbols[var]['value']

    def update(self, var, value):
        if var not in self.symbols:
            raise ValueError(f"Undeclared variable: {var}")
        if type(value).__name__ != self.symbols[var]['type']:
            raise ValueError(f"Type mismatch for {var}")
        self.symbols[var]['value'] = value

def semantic_analysis(ast, symbol_table):
    if isinstance(ast, NumberNode):
        return ast.value
    elif isinstance(ast, VarNode):
        return symbol_table.get(ast.name)
    elif isinstance(ast, AssignNode):
        value = semantic_analysis(ast.value, symbol_table)
        symbol_table.declare(ast.var, value)
        return value
    elif isinstance(ast, BinaryOpNode):
        left = semantic_analysis(ast.left, symbol_table)
        right = semantic_analysis(ast.right, symbol_table)
        if not (isinstance(left, (int, float)) and isinstance(right, (int, float))):
            raise ValueError("Type mismatch in binary operation")
        return left
    elif isinstance(ast, FunctionNode):
        arg = semantic_analysis(ast.arg, symbol_table)
        if not isinstance(arg, (int, float)):
            raise ValueError(f"Invalid argument type for {ast.func}")
        return arg
    elif isinstance(ast, IfNode):
        condition = semantic_analysis(ast.condition, symbol_table)
        if not isinstance(condition, (int, float)):
            raise ValueError("Condition must be numeric")
        return semantic_analysis(ast.body, symbol_table)

# ---- Bytecode Generation ----
class Bytecode:
    def __init__(self):
        self.instructions = []

    def emit(self, instr):
        self.instructions.append(instr)

def generate_bytecode(ast, bytecode, symbol_table):
    if isinstance(ast, NumberNode):
        bytecode.emit(('PUSH', ast.value))
    elif isinstance(ast, VarNode):
        bytecode.emit(('LOAD', ast.name))
    elif isinstance(ast, AssignNode):
        generate_bytecode(ast.value, bytecode, symbol_table)
        bytecode.emit(('STORE', ast.var))
    elif isinstance(ast, BinaryOpNode):
        generate_bytecode(ast.left, bytecode, symbol_table)
        generate_bytecode(ast.right, bytecode, symbol_table)
        op_map = {'+': 'ADD', '-': 'SUB', '*': 'MUL', '/': 'DIV', '^': 'POW', '>': 'GT', '<': 'LT', '==': 'EQ'}
        bytecode.emit((op_map[ast.op],))
    elif isinstance(ast, FunctionNode):
        generate_bytecode(ast.arg, bytecode, symbol_table)
        bytecode.emit(('CALL', ast.func))
    elif isinstance(ast, IfNode):
        generate_bytecode(ast.condition, bytecode, symbol_table)
        false_label = len(bytecode.instructions)
        bytecode.emit(('JUMP_IF_FALSE', None))  # Placeholder
        generate_bytecode(ast.body, bytecode, symbol_table)
        bytecode.instructions[false_label] = ('JUMP_IF_FALSE', len(bytecode.instructions))

# ---- Execution Engine ----
class VM:
    def __init__(self, symbol_table):
        self.stack = []
        self.symbol_table = symbol_table
        self.pc = 0

    def run(self, bytecode):
        self.pc = 0
        while self.pc < len(bytecode.instructions):
            instr = bytecode.instructions[self.pc]
            if instr[0] == 'PUSH':
                self.stack.append(instr[1])
            elif instr[0] == 'LOAD':
                self.stack.append(self.symbol_table.get(instr[1]))
            elif instr[0] == 'STORE':
                value = self.stack.pop()
                self.symbol_table.update(instr[1], value)
            elif instr[0] == 'ADD':
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a + b)
            elif instr[0] == 'SUB':
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a - b)
            elif instr[0] == 'MUL':
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a * b)
            elif instr[0] == 'DIV':
                b, a = self.stack.pop(), self.stack.pop()
                if b == 0:
                    raise ValueError("Division by zero")
                self.stack.append(a / b)
            elif instr[0] == 'POW':
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(a ** b)
            elif instr[0] == 'GT':
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1.0 if a > b else 0.0)
            elif instr[0] == 'LT':
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1.0 if a < b else 0.0)
            elif instr[0] == 'EQ':
                b, a = self.stack.pop(), self.stack.pop()
                self.stack.append(1.0 if a == b else 0.0)
            elif instr[0] == 'CALL':
                arg = self.stack.pop()
                if instr[1] == 'sin':
                    self.stack.append(math.sin(arg))
                elif instr[1] == 'cos':
                    self.stack.append(math.cos(arg))
                elif instr[1] == 'tan':
                    self.stack.append(math.tan(arg))
                elif instr[1] == 'sqrt':
                    self.stack.append(math.sqrt(arg))
                elif instr[1] == 'log':
                    self.stack.append(math.log10(arg))
                elif instr[1] == 'ln':
                    self.stack.append(math.log(arg))
            elif instr[0] == 'JUMP_IF_FALSE':
                if self.stack.pop() == 0.0:
                    self.pc = instr[1]
                    continue
            self.pc += 1
        return self.stack[-1] if self.stack else None

# ---- History File Management ----
def save_history(calc_name, history):
    """Save history to a .txt file."""
    if not os.path.exists(CALCULATOR_DIR):
        os.makedirs(CALCULATOR_DIR)
    file_path = os.path.join(CALCULATOR_DIR, f"{calc_name}_history.txt")
    with open(file_path, 'w') as f:
        for expr, res in history:
            f.write(f"{expr} = {res}\n")

def load_history(calc_name):
    """Load history from a .txt file."""
    file_path = os.path.join(CALCULATOR_DIR, f"{calc_name}_history.txt")
    history = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if '=' in line:
                    expr, res = line.strip().split(' = ', 1)
                    try:
                        res = float(res)
                        history.append((expr, res))
                    except ValueError:
                        continue
    return history

def view_history_file(calc_name):
    """Display the contents of a calculator's history .txt file."""
    file_path = os.path.join(CALCULATOR_DIR, f"{calc_name}_history.txt")
    if not os.path.exists(file_path):
        console.print(f"[yellow]No history file found for {calc_name}[/yellow]")
        return
    history_data = []
    with open(file_path, 'r') as f:
        for line in f:
            if '=' in line:
                expr, res = line.strip().split(' = ', 1)
                try:
                    res = float(res)
                    history_data.append((expr, res))
                except ValueError:
                    continue
    if not history_data:
        console.print(f"[yellow]History file for {calc_name} is empty[/yellow]")
        return
    table = Table(title=f"History for {calc_name}", border_style="cyan")
    table.add_column("Expression", style="magenta")
    table.add_column("Result", style="green")
    for expr, res in history_data:
        table.add_row(expr, str(res))
    console.print(table)

def delete_history(calc_name):
    """Delete a calculator's history file."""
    file_path = os.path.join(CALCULATOR_DIR, f"{calc_name}_history.txt")
    if os.path.exists(file_path):
        os.remove(file_path)

def list_calculators():
    """List all saved calculator history files."""
    if not os.path.exists(CALCULATOR_DIR):
        return []
    return [f.replace('_history.txt', '') for f in os.listdir(CALCULATOR_DIR) if f.endswith('_history.txt')]

# ---- UI and Main Loop ----
def display_menu():
    console.print(Panel(
        Text("Calculator Menu", style="bold cyan", justify="center"),
        border_style="green",
        padding=(1, 2)
    ))
    console.print("[bold magenta]Options:[/bold magenta]")
    console.print("1. Create A Calculator")
    console.print("2. Show Calculator")
    console.print("3. Select Calculator")
    console.print("4. Delete Calculator")
    console.print("5. View Calculator History")
    console.print("6. EXIT")
    console.print("\n[bold yellow]Enter choice (1-6):[/bold yellow]")

def display_welcome(calc_name):
    console.print(Panel(
        Text(f"Calculator: {calc_name}", style="bold cyan", justify="center"),
        border_style="green",
        padding=(1, 2)
    ))
    console.print("[bold magenta]Supported operations:[/bold magenta]")
    console.print("- Basic: +, -, *, /, ^ (power)")
    console.print("- Scientific: sin, cos, tan, sqrt, log, ln, pi, e")
    console.print("- Variables: x = 3 + 4")
    console.print("- Control: if (x > 0) { x = x - 1 }")
    console.print("- Commands: history, clear, end, exit")
    console.print("\n[bold yellow]Enter expression or command:[/bold yellow]")

def display_history():
    if not history:
        console.print("[yellow]No calculations in history[/yellow]")
        return
    table = Table(title="Calculation History", border_style="cyan")
    table.add_column("Expression", style="magenta")
    table.add_column("Result", style="green")
    for expr, res in history:
        table.add_row(expr, str(res))
    console.print(table)

@bindings.add(Keys.ControlC)
def _(event):
    console.print("[bold red]Exiting calculator...[/bold red]")
    event.app.exit()

def calculator_session(calc_name):
    global history
    symbol_table = SymbolTable()
    bytecode = Bytecode()
    vm = VM(symbol_table)
    history = load_history(calc_name)  # Load existing history
    display_welcome(calc_name)
    while True:
        try:
            expr = session.prompt()
            if not expr.strip():
                continue
            if expr.lower() == 'history':
                display_history()
                continue
            elif expr.lower() == 'clear':
                console.clear()
                display_welcome(calc_name)
                continue
            elif expr.lower() == 'exit':
                console.print("[bold red]Exiting calculator...[/bold red]")
                save_history(calc_name, history)
                break
            elif expr.lower() == 'end':
                console.print("[bold yellow]End session. Delete history? (y/n): [/bold yellow]")
                choice = session.prompt().strip().lower()
                save_history(calc_name, history)
                if choice == 'y':
                    delete_history(calc_name)
                    console.print(f"[green]History for {calc_name} deleted[/green]")
                history = []
                break
            tokens = tokenize(expr)
            parser = Parser(tokens)
            ast = parser.statement()
            semantic_analysis(ast, symbol_table)
            bytecode.instructions.clear()
            generate_bytecode(ast, bytecode, symbol_table)
            result = vm.run(bytecode)
            if result is not None:
                console.print(f"[bold green]Result: {result}[/bold green]")
                history.append((expr, result))
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
        except KeyboardInterrupt:
            console.print("[bold red]Exiting calculator...[/bold red]")
            save_history(calc_name, history)
            break
    return True  # Return to main menu

def main():
    while True:
        display_menu()
        try:
            choice = menu_session.prompt().strip()
            if choice == '1':
                console.print("[bold yellow]Enter calculator name: [/bold yellow]")
                calc_name = session.prompt().strip()
                if not calc_name:
                    console.print("[bold red]Name cannot be empty[/bold red]")
                    continue
                if calc_name in list_calculators():
                    console.print("[bold red]Calculator already exists[/bold red]")
                    continue
                calculator_session(calc_name)
            elif choice == '2':
                calculators = list_calculators()
                if not calculators:
                    console.print("[yellow]No calculators found[/yellow]")
                else:
                    console.print("[bold magenta]Saved Calculators:[/bold magenta]")
                    for calc in calculators:
                        console.print(f"- {calc}")
            elif choice == '3':
                calculators = list_calculators()
                if not calculators:
                    console.print("[yellow]No calculators to select[/yellow]")
                    continue
                console.print("[bold yellow]Select calculator:[/bold yellow]")
                for i, calc in enumerate(calculators, 1):
                    console.print(f"{i}. {calc}")
                try:
                    index = int(session.prompt().strip()) - 1
                    if 0 <= index < len(calculators):
                        calculator_session(calculators[index])
                    else:
                        console.print("[bold red]Invalid selection[/bold red]")
                except ValueError:
                    console.print("[bold red]Enter a number[/bold red]")
            elif choice == '4':
                calculators = list_calculators()
                if not calculators:
                    console.print("[yellow]No calculators to delete[/yellow]")
                    continue
                console.print("[bold yellow]Select calculator to delete:[/bold yellow]")
                for i, calc in enumerate(calculators, 1):
                    console.print(f"{i}. {calc}")
                try:
                    index = int(session.prompt().strip()) - 1
                    if 0 <= index < len(calculators):
                        calc_name = calculators[index]
                        console.print(f"[bold yellow]Confirm deletion of {calc_name}? (y/n): [/bold yellow]")
                        if session.prompt().strip().lower() == 'y':
                            delete_history(calc_name)
                            console.print(f"[green]{calc_name} deleted[/green]")
                        else:
                            console.print("[yellow]Deletion cancelled[/yellow]")
                    else:
                        console.print("[bold red]Invalid selection[/bold red]")
                except ValueError:
                    console.print("[bold red]Enter a number[/bold red]")
            elif choice == '5':
                calculators = list_calculators()
                if not calculators:
                    console.print("[yellow]No calculators to view[/yellow]")
                    continue
                console.print("[bold yellow]Select calculator to view history:[/bold yellow]")
                for i, calc in enumerate(calculators, 1):
                    console.print(f"{i}. {calc}")
                try:
                    index = int(session.prompt().strip()) - 1
                    if 0 <= index < len(calculators):
                        view_history_file(calculators[index])
                    else:
                        console.print("[bold red]Invalid selection[/bold red]")
                except ValueError:
                    console.print("[bold red]Enter a number[/bold red]")
            elif choice == '6':
                console.print("[bold red]Exiting program...[/bold red]")
                break
            else:
                console.print("[bold red]Invalid choice. Enter 1-6.[/bold red]")
        except KeyboardInterrupt:
            console.print("[bold red]Exiting program...[/bold red]")
            break

# ---- Unit Tests ----
class TestCompilerCalculator(unittest.TestCase):
    def test_tokenize(self):
        tokens = tokenize("x = 3 + 4")
        self.assertEqual(tokens[0], Token('IDENTIFIER', 'x'))
        self.assertEqual(tokens[1], Token('ASSIGNMENT', '='))
        self.assertEqual(tokens[2], Token('NUMBER', 3.0))

    def test_parser(self):
        tokens = tokenize("3 + 4 * 2")
        parser = Parser(tokens)
        ast = parser.expression()
        self.assertIsInstance(ast, BinaryOpNode)
        self.assertEqual(ast.op, '+')

    def test_semantic_analysis(self):
        symbol_table = SymbolTable()
        tokens = tokenize("x = 3")
        parser = Parser(tokens)
        ast = parser.statement()
        semantic_analysis(ast, symbol_table)
        self.assertEqual(symbol_table.get('x'), 3.0)

    def test_bytecode_execution(self):
        symbol_table = SymbolTable()
        bytecode = Bytecode()
        vm = VM(symbol_table)
        tokens = tokenize("x = 3 + 4")
        parser = Parser(tokens)
        ast = parser.statement()
        generate_bytecode(ast, bytecode, symbol_table)
        result = vm.run(bytecode)
        self.assertEqual(symbol_table.get('x'), 7.0)

    def test_if_statement(self):
        symbol_table = SymbolTable()
        bytecode = Bytecode()
        vm = VM(symbol_table)
        tokens = tokenize("x = 5 if (x > 0) { x = x - 1 }")
        parser = Parser(tokens)
        ast = parser.statement()
        semantic_analysis(ast, symbol_table)
        generate_bytecode(ast, bytecode, symbol_table)
        result = vm.run(bytecode)
        self.assertEqual(symbol_table.get('x'), 4.0)

    def test_history_save_load(self):
        calc_name = "test_calc"
        history = [("3 + 4", 7.0), ("x = 5", 5.0)]
        save_history(calc_name, history)
        loaded_history = load_history(calc_name)
        self.assertEqual(loaded_history, history)
        delete_history(calc_name)

    def test_delete_calculator(self):
        calc_name = "test_calc"
        save_history(calc_name, [("3 + 4", 7.0)])
        self.assertTrue(os.path.exists(os.path.join(CALCULATOR_DIR, f"{calc_name}_history.txt")))
        delete_history(calc_name)
        self.assertFalse(os.path.exists(os.path.join(CALCULATOR_DIR, f"{calc_name}_history.txt")))

if __name__ == "__main__":
    main()