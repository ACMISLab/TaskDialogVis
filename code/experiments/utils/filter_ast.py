from lark import Lark, Transformer, Tree

filter_grammar_corrected_v4 = r"""
    ?start: expr
    ?expr : or_expr
    ?or_expr  : and_expr ( "||" and_expr )+ -> or_op
              | and_expr
    ?and_expr : factor ( "&&" factor )+   -> and_op
              | factor
    ?factor   : comparison
              | atom          // An atom can now be a function call or key access
              | "(" expr ")"

    comparison: atom OPARATOR atom -> comparison_op // Operands can now be function calls, key access, etc.

    ?atom     : function_call  // Function calls
              | key_access     // NEW: Key access like ['key']
              | IDENTIFIER -> identifier
              | value

    // Rule for function calls: IDENTIFIER followed by () potentially containing arguments
    function_call: IDENTIFIER "(" ( expr ( "," expr )* )? ")" -> func_call

    // NEW: Rule for accessing keys using brackets like ['key']
    key_access: "[" STRING "]" -> key_access_op


    ?value: NUMBER -> number
          | STRING -> string
          | BOOL   -> boolean

    OPARATOR: ">" | "<" | "==" | ">=" | "<=" | "!="

    // MODIFIED IDENTIFIER: Allows starting with digits if letters/underscores follow
    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+[a-zA-Z_]+[a-zA-Z0-9_]*/
    NUMBER: /\d+(\.\d+)?/           // Defined AFTER IDENTIFIER might sometimes help priority, but the regex distinction is safer
    STRING: /'[^']*'/ | /"[^"]*"/
    BOOL: "true" | "false"

    %import common.WS
    %ignore WS
"""

# Define the Transformer class (same as above)
# Note: You might want to add a method for 'key_access_op' in the Transformer
# if you need to transform the AST node for key access.
class NormalizeAST(Transformer):
    def and_op(self, items):
        # Assuming items are comparable, for sorting Tree nodes can be compared by their pretty() string representation
        # Or you might define a specific comparison logic based on the node type and value
        sorted_items = sorted(items, key=lambda item: item.pretty())
        return Tree('and_op', sorted_items)

    def or_op(self, items):
         # Assuming items are comparable, for sorting Tree nodes can be compared by their pretty() string representation
        sorted_items = sorted(items, key=lambda item: item.pretty())
        return Tree('or_op', sorted_items)

    # Add a transformation for the new key_access rule if needed
    # def key_access_op(self, items):
    #     # items will contain the STRING token
    #     return Tree('key_access_op', items)


from lark import Lark, Transformer, Tree, Token

def filter_to_ast(filter_str):
    # Keep the existing replacements
    filter_str = filter_str.replace("!==", "!=").replace("datum.","").replace("===","==").replace("datum['","").replace("']","")
    try:
        # Use the updated grammar
        parser_corrected = Lark(filter_grammar_corrected_v4, start='start')
        tree = parser_corrected.parse(filter_str)
        # Apply the transformer to normalize the tree
        normalized_tree = NormalizeAST().transform(tree)
        return normalized_tree
    except Exception as e:
        print(f"Error creating parser or parsing: {e}")
        return None

# Test cases
a_norm = filter_to_ast("datum.Year >= 2020")
b_norm = filter_to_ast(" datum.Year >= 2020")
# Add a test case for the bracket syntax
c_norm = filter_to_ast("datum['24h_High_USD'] > 8000")
d_norm = filter_to_ast("datum['Some_Other_Key'] == 'value'")
e_norm = filter_to_ast("datum['Numeric_Key'] > 100") # Test if STRING rule handles numbers in quotes


print("\nNormalized AST for string a:")
print(a_norm.pretty() if a_norm else "Error")

print("\nNormalized AST for string b:")
print(b_norm.pretty() if b_norm else "Error")

print(f"\nAre normalized ASTs for a and b equal? {a_norm == b_norm}")


print("\nNormalized AST for string c:")
print(c_norm.pretty() if c_norm else "Error")

print("\nNormalized AST for string d:")
print(d_norm.pretty() if d_norm else "Error")

print("\nNormalized AST for string e:")
print(e_norm.pretty() if e_norm else "Error")