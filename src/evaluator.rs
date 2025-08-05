use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::ast::{ArrayLiteral, AssignmentExpr, BinaryExpr, BooleanLiteral, CallExpr, ForStatement, FunctionDeclaration, IdentifierExpr, IfStatement, ImportAst, LiteralExpr, MemberExpr, NodeType, ObjectLiteral, Program, Property, Stmt, StringVal, VariableDeclaration};
use crate::environment::Environment;
use crate::mk_null;
use crate::runtime::RuntimeVal;
use crate::types::{ArrayVal, BooleanVal, FunctionVal, IntegerVal, NativeFnValue, NullVal, ObjectVal, ValueType, RESERVED_NAMES};
use crate::types::ValueType::{Array, Boolean, Function, Integer, Null, Object};

pub fn eval(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    if let Ok(program) = ast_node.clone().downcast::<Program>() {
        let mut has_main_function = false;

        for stmt in &program.body {
            match stmt.kind() {
                NodeType::FunctionDeclaration => {
                    if let Some(func_decl) = stmt.as_any().downcast_ref::<FunctionDeclaration>() {
                        if func_decl.name == "main" {
                            has_main_function = true;
                        }
                    }
                    evaluate(stmt.clone(), env);
                },
                NodeType::VariableDeclaration => {
                    evaluate(stmt.clone(), env);
                },
                NodeType::ImportAst => {
                    evaluate(stmt.clone(), env);
                },
                _ => {}
            }
        }

        if has_main_function {
            let main_identifier = Box::new(IdentifierExpr {
                kind: NodeType::Identifier,
                name: "main".to_string(),
            });

            let main_call = Box::new(CallExpr {
                kind: NodeType::CallExpr,
                caller: main_identifier,
                args: Vec::new(),
            });

            return evaluate(main_call, env);
        } else {
            let mut last_evaluated: Box<dyn RuntimeVal + Send + Sync> = mk_null();

            for stmt in program.body.into_iter().filter(|s| {
                !matches!(s.kind(),
                    NodeType::FunctionDeclaration |
                    NodeType::VariableDeclaration |
                    NodeType::ImportAst
                ) && !(matches!(s.kind(), NodeType::Identifier) &&
                    s.as_any().downcast_ref::<IdentifierExpr>().unwrap().name == "null")
            }) {
                last_evaluated = evaluate(stmt, env);
            }

            return last_evaluated;
        }
    }

    evaluate(ast_node, env)
}
pub fn evaluate(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {

    match ast_node.kind() {
        NodeType::NumericLiteral => {
            let literal = ast_node.as_any().downcast_ref::<LiteralExpr>()
                .expect("Expected a LiteralExpr");
            Box::from(IntegerVal {
                r#type: Some(Integer),
                value: literal.value,
            })
        },
        NodeType::BooleanLiteral => {
            let bool_node = ast_node.as_any().downcast_ref::<BooleanLiteral>().unwrap();
            Box::from(BooleanVal { r#type: Some(Boolean), value: bool_node.value })
        },
        NodeType::NullLiteral => {
            mk_null()
        },
        NodeType::BinaryExpression => {
            eval_binary_expr(ast_node, env)
        },
        NodeType::Program => {
            eval_program(ast_node, env)
        },
        NodeType::Identifier => {
            eval_identifier(ast_node, env)
        },
        NodeType::VariableDeclaration => {
            eval_var_declaration(ast_node, env)
        },
        NodeType::StringLiteral => {
            let str = ast_node.as_any().downcast_ref::<StringVal>()
                .expect("Expected a LiteralExpr");
            Box::from(StringVal {
                r#type: Some(ValueType::String),
                kind: NodeType::StringLiteral,
                value: str.value.clone(),
            })
        },
        NodeType::AssignmentExpr => {
            eval_assignment(ast_node, env)
        },
        NodeType::ObjectLiteral => {
            eval_object_expr(ast_node, env)
        },
        NodeType::CallExpr => {
            eval_call_expr(ast_node, env)
        },
        NodeType::FunctionDeclaration => {
            eval_function_declaration(ast_node, env)
        },
        NodeType::MemberExpr => {
            eval_member_expr(ast_node, env)
        },
        NodeType::Property => panic!("This ast node has not yet been setup for interpretation {:#?}", ast_node),
        NodeType::ArrayLiteral => {
            eval_array_expr(ast_node, env)
        },
        NodeType::ImportAst => {
            let import_ast = ast_node.as_any().downcast_ref::<ImportAst>()
                .expect("Expected ImportAst node");
            let mut last: Box<dyn RuntimeVal + Send + Sync> = mk_null();
            for stmt in import_ast.body.iter() {
                last = evaluate(stmt.clone(), env);
            }
            last

        },
        NodeType::IfStatement => eval_if_statement(ast_node, env),
        NodeType::ForStatement => eval_for_statement(ast_node, env),
    }
}

pub fn eval_for_statement(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let for_stmt = ast_node
        .as_any()
        .downcast_ref::<ForStatement>()
        .expect("Expected a ForStatement node");

    if let Some(init_stmt) = &for_stmt.initializer {
        evaluate(init_stmt.clone(), env);
    }

    let mut last: Box<dyn RuntimeVal + Send + Sync> = mk_null();

    loop {
        if let Some(cond_stmt) = &for_stmt.condition {
            let cond_val = evaluate(cond_stmt.clone(), env);
            let keep_going = cond_val
                .as_any()
                .downcast_ref::<BooleanVal>()
                .map(|b| b.value)
                .unwrap_or_else(|| panic!("Condition not Boolean"));
            if !keep_going {
                break;
            }
        }

        for stmt in &for_stmt.body {
            let val = evaluate(stmt.clone(), env);
            //println!("DEBUG ─ evaluate renvoie : {:?}", val);
            last = val;
        }


        if let Some(inc_stmt) = &for_stmt.increment {
            evaluate(inc_stmt.clone(), env);
        }
    }

    last
}


pub fn eval_if_statement(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let if_stmt = ast_node.as_any().downcast_ref::<IfStatement>()
        .expect("Expected an IfStatement node");

    //println!("{:?}", if_stmt);

    let cond_val = evaluate(if_stmt.condition.clone(), env);
    let condition_is_true = if let Some(boolean) = cond_val.as_any().downcast_ref::<BooleanVal>() {
        boolean.value
    } else {
        panic!("If condition is not a boolean : {:?}", cond_val);
    };

    if condition_is_true {
        let mut then_closure = || {
            let mut last: Box<dyn RuntimeVal + Send + Sync> = mk_null();
            for stmt in if_stmt.then_branch.iter() {
                last = evaluate(stmt.clone(), env);
            }
            last
        };
        then_closure()
    } else if let Some(else_branch) = &if_stmt.else_branch {
        let mut else_closure = || {
            let mut last: Box<dyn RuntimeVal + Send + Sync> = mk_null();
            for stmt in else_branch.iter() {
                last = evaluate(stmt.clone(), env);
            }
            last
        };
        else_closure()
    } else {
        mk_null()
    }
}


pub fn eval_array_expr(node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let array_node = node.downcast::<ArrayLiteral>()
        .expect("Expected an ArrayLiteral node");

    let elements: Vec<Box<dyn RuntimeVal + Send + Sync>> = array_node
        .elements
        .into_iter()
        .map(|expr| evaluate(expr, env))
        .collect();

    Box::from(ArrayVal {
        r#type: Option::from(Array),
        elements: Arc::new(Mutex::new(elements)),
    })
}



pub fn eval_assignment(node: Box<dyn Stmt>, env: &mut Environment, ) -> Box<dyn RuntimeVal + Send + Sync> {
    let assignment = node
        .downcast::<AssignmentExpr>()
        .expect("Expected an AssignmentExpr node");

    let new_value = evaluate(
        assignment.value.expect("Assignment missing RHS"),
        env,
    );

    if let Some(ident) = assignment.assigne.as_any().downcast_ref::<IdentifierExpr>() {
        let var_name = ident.name.clone();

        let result = env.set_var(var_name, new_value.clone(), None);

        result
    }
    else if let Some(member) = assignment.assigne.as_any().downcast_ref::<MemberExpr>() {
        let object_val = evaluate(member.object.clone(), env);

        if let Some(obj) = object_val.as_any().downcast_ref::<ObjectVal>() {
            let prop_name = if let Some(ident) =
                member.property.as_any().downcast_ref::<IdentifierExpr>()
            {
                ident.name.clone()
            } else {
                panic!("Member expression expected an identifier for non-computed property");
            };
            obj.properties.lock().unwrap().insert(prop_name, new_value.clone());
            new_value
        }
        else if let Some(arr) = object_val.as_any().downcast_ref::<ArrayVal>() {
            let index_val = evaluate(member.property.clone(), env);
            if let Some(num) = index_val.as_any().downcast_ref::<IntegerVal>() {
                let index = num.value as usize;
                let mut elems = arr.elements.lock().unwrap();
                if index < elems.len() {
                    elems[index] = new_value.clone();
                    new_value
                } else {
                    panic!(
                        "Index out of bounds: {} is greater than the array size {}",
                        index,
                        elems.len()
                    );
                }
            } else {
                panic!("The index of an array must be a number");
            }
        }
        else {
            panic!("Assignment target is not an object or an array");
        }
    }
    else {
        panic!("Invalid LHS in assignment: {:?}", assignment.assigne);
    }
}



pub fn eval_var_declaration(declaration: Box<dyn Stmt>, env: &mut Environment, ) -> Box<dyn RuntimeVal + Send + Sync> {
    let var_declaration = declaration
        .downcast::<VariableDeclaration>()
        .expect("Expected a VariableDeclaration node");

    let declared_type = var_declaration.r#type.clone();

    let value = match var_declaration.value {
        Some(expr) => evaluate(expr, env),
        None => mk_null(),
    };

    env.set_var(var_declaration.name.clone(), value.clone(), declared_type)
}
pub fn eval_member_expr(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let member = ast_node.downcast::<MemberExpr>()
        .expect("Expected a MemberExpr node");

    let object_val = evaluate(member.object, env);

    if let Some(obj) = object_val.as_any().downcast_ref::<ObjectVal>() {
        let prop_name = if let Some(ident) = member.property.as_any().downcast_ref::<IdentifierExpr>() {
            ident.name.clone()
        } else {
            panic!("Member expression expected an identifier for a non-computed property");
        };

        match obj.properties.lock().unwrap().get(&prop_name) {
            Some(val) => val.clone(),
            None => panic!("The property '{}' does not exist in the object", prop_name),
        }
    } else if let Some(arr) = object_val.as_any().downcast_ref::<ArrayVal>() {
        let index_val = evaluate(member.property, env);
        if let Some(num) = index_val.as_any().downcast_ref::<IntegerVal>() {
            let index = num.value as usize;
            if index < arr.elements.lock().unwrap().len() {
                arr.elements.lock().unwrap()[index].clone()
            } else {
                panic!("Index out of bounds: {} is greater than the array size {}", index, arr.elements.lock().unwrap().len());
            }
        } else {
            panic!("The index of an array must be a number");
        }
    } else {
        panic!("Member access on a type that is neither an object nor an array");
    }
}

pub fn eval_binary_expr(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let binary_expr = ast_node.downcast::<BinaryExpr>()
        .expect("Expected a BinaryExpr node");

    let lhs = evaluate(binary_expr.left, env);
    let rhs = evaluate(binary_expr.right, env);

    if let (Some(lhs_num), Some(rhs_num)) = (lhs.as_any().downcast_ref::<IntegerVal>(), rhs.as_any().downcast_ref::<IntegerVal>()) {
        return eval_numeric_binary_expr(
            lhs_num,
            rhs_num,
            &binary_expr.operator,
        );
    } else if let (Some(lhs_str), Some(rhs_str)) = (lhs.as_any().downcast_ref::<StringVal>(), rhs.as_any().downcast_ref::<StringVal>()) {
        return eval_string_binary_expr(
            lhs_str,
            rhs_str,
            &binary_expr.operator,
        );
    }

    Box::from(NullVal {
        r#type: Some(Null),
    })
}

pub fn eval_string_binary_expr(lhs: &StringVal, rhs: &StringVal, operator: &str, ) -> Box<dyn RuntimeVal + Send + Sync> {
    let lhs_value = lhs.value.clone();
    let rhs_value = rhs.value.clone();

    //println!("l {}, r{}", lhs_value, rhs_value);

    match operator {
        "=="  => Box::from(BooleanVal { r#type: Option::from(Boolean), value: lhs_value.eq(&rhs_value)}),

        _ => panic!("Unknown binary operator: {}", operator),
    }
}

pub fn eval_numeric_binary_expr(lhs: &IntegerVal, rhs: &IntegerVal, operator: &str, ) -> Box<dyn RuntimeVal + Send + Sync> {
    let lhs_value = lhs.value;
    let rhs_value = rhs.value;

    match operator {
        "+" => Box::from(IntegerVal { r#type: Option::from(Integer), value: lhs_value + rhs_value }),
        "-" => Box::from(IntegerVal { r#type: Option::from(Integer), value: lhs_value - rhs_value }),
        "*" => Box::from(IntegerVal { r#type: Option::from(Integer), value: lhs_value * rhs_value }),
        "/" => {
            if rhs_value == 0.0 {
                panic!("Division par zéro interdite");
            }
            Box::from(IntegerVal { r#type: Option::from(Integer), value: lhs_value / rhs_value })
        },
        "%" => Box::from(IntegerVal { r#type: Option::from(Integer), value: lhs_value % rhs_value }),

        ">"  => Box::from(BooleanVal { r#type: Option::from(Boolean), value: lhs_value  > rhs_value }),
        "<"  => Box::from(BooleanVal { r#type: Option::from(Boolean), value: lhs_value  < rhs_value }),
        ">=" => Box::from(BooleanVal { r#type: Option::from(Boolean), value: lhs_value >= rhs_value }),
        "<=" => Box::from(BooleanVal { r#type: Option::from(Boolean), value: lhs_value <= rhs_value }),
        "!=" => Box::from(BooleanVal { r#type: Option::from(Boolean), value: lhs_value != rhs_value }),
        "=="  => Box::from(BooleanVal { r#type: Option::from(Boolean), value: lhs_value == rhs_value }),

        _ => panic!("Unknown binary operator: {}", operator),
    }
}

pub fn eval_program(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let program = ast_node.downcast::<Program>()
        .expect("Expected a Program node");

    let mut last_evaluated: Box<dyn RuntimeVal + Send + Sync> = mk_null();

    for stmt in program.body.into_iter().filter(|s| { !(matches!(s.kind(), NodeType::Identifier) && s.as_any().downcast_ref::<IdentifierExpr>().unwrap().name == "null") }) {
        last_evaluated = evaluate(stmt, env);
    }
    
    last_evaluated
}

pub fn eval_identifier(ast_node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let identifier_expr = ast_node.downcast::<IdentifierExpr>()
        .expect("Expected an IdentifierExpr node");

    if RESERVED_NAMES.contains(&identifier_expr.name) {
        return mk_null();
    }

    env.lookup_var(identifier_expr.name)
}


pub fn eval_object_expr(node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let obj = node
        .downcast::<ObjectLiteral>()
        .expect("Expected ObjectLiteral");

    let mut props: HashMap<String, Box<dyn RuntimeVal + Send + Sync>> = HashMap::new();

    for Property { key, value, .. } in obj.properties {
        let runtime_val = if let Some(expr) = value {
            evaluate(expr, env)
        } else {
            env.lookup_var(key.clone())
        };
        props.insert(key, runtime_val);
    }

    Box::from(ObjectVal {
        r#type: Option::from(Object),
        properties: Arc::new(Mutex::new(props)),
    })
}

pub fn eval_call_expr(node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync> {
    let call = node
        .downcast::<CallExpr>()
        .expect("Expected CallExpr");

    let args: Vec<Box<dyn RuntimeVal + Send + Sync>> = call
        .args
        .into_iter()
        .map(|arg| evaluate(arg, env))
        .collect();

    let callee = evaluate(call.caller, env);

    if let Some(native) = callee.as_any().downcast_ref::<NativeFnValue>() {
        return (native.call)(args, env);
    }

    if let Some(func) = callee.as_any().downcast_ref::<FunctionVal>() {
        let mut scope = Environment::new(Some(Box::new(env.clone())));

        if args.len() != func.parameters.len() {
            panic!(
                "Function `{}` expected {} arguments but got {}",
                func.name,
                func.parameters.len(),
                args.len()
            );
        }

        for (param, arg_val) in func.parameters.iter().zip(args.into_iter()) {
            scope.set_var(param.clone(), arg_val, None);
        }

        let mut result: Box<dyn RuntimeVal + Send + Sync> = mk_null();
        for stmt in func.body.iter() {
            result = evaluate(stmt.clone(), &mut scope);
        }

        return result;
    }

    panic!("Cannot call value that is not a function: {:?}", callee);
}

pub fn eval_function_declaration(node: Box<dyn Stmt>, env: &mut Environment) -> Box<dyn RuntimeVal + Send + Sync>{
    let func = node
        .downcast::<FunctionDeclaration>()
        .expect("Expected FunctionDeclaration");

    let function = FunctionVal{
        r#type: Option::from(Function),
        parameters: func.parameters,
        name: func.name.clone(),
        body: Arc::new(func.body),
        declaration_env: Arc::new(Mutex::new(env.clone())),
    };

    //let decl_env : &mut Environment = unsafe {&mut *function_val.declaration_env} ;

    env.set_var(func.name, Box::from(function), Option::from(Function))
}