# Advanced Terminal Calculator (Adv_Calc)
# Provides basic arithmetic, scientific functions, memory operations, and history with a styled interface

import math  # For mathematical functions (sin, cos, sqrt) and constants (pi, e)
from rich.console import Console  # For styled terminal output (colors, formatting)
from rich.panel import Panel  # For creating bordered panels (e.g., welcome message)
from rich.table import Table  # For displaying calculation history in a table
from rich.text import Text  # For styled text (bold, colored) in output
from prompt_toolkit import PromptSession  # For handling user input with a prompt
from prompt_toolkit.key_binding import KeyBindings  # For binding keys (e.g., Ctrl+C)
from prompt_toolkit.keys import Keys  # For defining key constants (e.g., ControlC)

# Initialize global objects
console = Console()  # Console object for styled output
session = PromptSession(">>> ", multiline=False)  # Input prompt for user expressions
bindings = KeyBindings()  # Key bindings for custom actions (e.g., exit on Ctrl+C)
memory = 0  # Stores memory value for M+, M-, MR, MC operations
history = []  # List of (expression, result) tuples for calculation history

def display_welcome():
    """Display a styled welcome panel with supported operations."""
    console.print(Panel(
        Text("Advanced Terminal Calculator", style="bold cyan", justify="center"),
        border_style="green",
        padding=(1, 2)
    ))
    console.print("[bold magenta]Supported operations:[/bold magenta]")
    console.print("- Basic: +, -, *, /, ^ (power), % (modulo)")
    console.print("- Scientific: sin, cos, tan, sqrt, log, ln, pi, e")
    console.print("- Memory: M+ (add to memory), M- (subtract from memory), MR (recall), MC (clear)")
    console.print("- Other: history (view past calculations), clear, exit")
    console.print("\n[bold yellow]Enter expression or command:[/bold yellow]")

def evaluate_expression(expr):
    """Evaluate user input (expression or command) and return result or handle actions."""
    global memory  # Access and modify memory
    try:
        # Replace user-friendly operators and functions for eval
        expr = expr.replace('^', '**').replace('pi', str(math.pi)).replace('e', str(math.e))
        expr = expr.replace('sin(', 'math.sin(').replace('cos(', 'math.cos(').replace('tan(', 'math.tan(')
        expr = expr.replace('sqrt(', 'math.sqrt(').replace('log(', 'math.log10(').replace('ln(', 'math.log(')
        
        # Handle special commands
        if expr.lower() == 'history':
            display_history()  # Show calculation history
            return None
        elif expr.lower() == 'clear':
            console.clear()  # Clear terminal
            display_welcome()  # Redisplay welcome panel
            return None
        elif expr.lower() == 'exit':
            console.print("[bold red]Exiting calculator...[/bold red]")
            return False  # Signal to exit program
        elif expr.lower() == 'm+':
            memory += result if 'result' in locals() else 0  # Add last result to memory
            console.print(f"[green]Memory updated: {memory}[/green]")
            return None
        elif expr.lower() == 'm-':
            memory -= result if 'result' in locals() else 0  # Subtract last result from memory
            console.print(f"[green]Memory updated: {memory}[/green]")
            return None
        elif expr.lower() == 'mr':
            console.print(f"[green]Memory recalled: {memory}[/green]")
            return memory  # Return memory value
        elif expr.lower() == 'mc':
            memory = 0  # Clear memory
            console.print("[green]Memory cleared[/green]")
            return None
        
        # Evaluate mathematical expression
        result = eval(expr, {"math": math, "memory": memory})
        history.append((expr, result))  # Log calculation to history
        console.print(f"[bold green]Result: {result}[/bold green]")
        return result
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")  # Display error for invalid input
        return None

def display_history():
    """Display a table of past calculations from history."""
    if not history:
        console.print("[yellow]No calculations in history[/yellow]")
        return
    table = Table(title="Calculation History", border_style="cyan")
    table.add_column("Expression", style="magenta")
    table.add_column("Result", style="green")
    for expr, res in history:
        table.add_row(expr, str(res))  # Add each calculation as a row
    console.print(table)

# Bind Ctrl+C to exit program
@bindings.add(Keys.ControlC)
def _(event):
    """Handle Ctrl+C to exit gracefully."""
    console.print("[bold red]Exiting calculator...[/bold red]")
    event.app.exit()

def main():
    """Main program loop to run the calculator."""
    display_welcome()  # Show initial welcome panel
    while True:
        try:
            expr = session.prompt()  # Get user input
            if not expr.strip():
                continue  # Skip empty input
            result = evaluate_expression(expr)  # Process input
            if result is False:
                break  # Exit if evaluate_expression returns False
        except KeyboardInterrupt:
            console.print("[bold red]Exiting calculator...[/bold red]")
            break  # Handle Ctrl+C

if __name__ == "__main__":
    main()  # Start the calculator