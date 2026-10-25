"""Small Tier 2 reporting helpers."""

def iut_p(component_ps: dict) -> dict:
    if not component_ps: raise ValueError("at least one component p-value is required")
    bad={k:v for k,v in component_ps.items() if not 0 <= float(v) <= 1}
    if bad: raise ValueError(f"invalid p-values: {bad}")
    binding=max(component_ps,key=lambda k:float(component_ps[k]))
    return {"components":dict(component_ps),"p_composite":float(component_ps[binding]),"binding_component":binding}
