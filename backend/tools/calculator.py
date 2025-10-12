import ast
import operator as op

# Allowed operators for safe evaluation
_ALLOWED_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _safe_eval_node(node, depth=0):
    if depth > 20:
        raise ValueError("Expression too complex")

    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body, depth + 1)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed")

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        operand = _safe_eval_node(node.operand, depth + 1)
        return _ALLOWED_UNARY_OPS[type(node.op)](operand)

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        left = _safe_eval_node(node.left, depth + 1)
        right = _safe_eval_node(node.right, depth + 1)
        # Guard against excessively large exponentiation
        if isinstance(node.op, ast.Pow):
            if abs(left) > 1e6 or abs(right) > 12:
                raise ValueError("Exponent too large")
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)

    raise ValueError("Unsupported expression")


def calculate(expression: str):
    """Safely evaluate a basic math expression.

    Supports +, -, *, /, %, **, parentheses, and unary +/-.
    Also treats '^' as exponent for convenience.
    """
    try:
        expr = (expression or "").strip()
        if not expr:
            return {"error": "Empty expression"}
        # Normalize caret exponent to Python exponent
        expr = expr.replace("^", "**")
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval_node(tree)
        # Normalize -0.0 to 0.0
        if isinstance(result, float) and result == 0:
            result = 0.0
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
