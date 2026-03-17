import os

def format_solution_impl(solution, args: dict) -> str:
    """Format metrics and visualization output for a completed model run."""
    # Calculate metrics
    try:
        total_acfs = sum(solution.ACFEstablishment_x_wi[0])
        total_assign = sum(
            solution.LandRescueVehicle_thetaVar_wim[0][i][m]
            for i in range(len(solution.LandRescueVehicle_thetaVar_wim[0]))
            for m in range(len(solution.LandRescueVehicle_thetaVar_wim[0][i]))
        )
        cost = solution.TotalCost
    except Exception:
        total_acfs = None
        total_assign = None
        cost = None

    metrics = ""
    if total_acfs is not None:
        metrics += f"**📊 Solution Metrics:**\n"
        metrics += f"- **Total Resources Assigned to ACFs:** {total_assign}\n"
        metrics += f"- **Total Established Temporary Hospitals (ACFs):** {total_acfs}\n"
        metrics += f"- **Total Cost:** ${cost:,.0f}\n"
        metrics += "---\n"
    else:
        metrics += "**📊 Solution Metrics:** Detailed metrics saved to output directory.\n---\n"

    # Images
    instance = args['Instance']
    model = args['Model']
    solver = args['Solver']
    base_path = os.path.join('UI', 'Solution_UI')
    img_all = os.path.join(base_path, f"{instance}_{model}_{solver}_all_facilities.png")
    img_open = os.path.join(base_path, f"{instance}_{model}_{solver}_open_acfs_only.png")

    images_section = ""
    if os.path.exists(img_all) and os.path.exists(img_open):
        images_section += f"![All Facilities]({img_all})  ![Open ACFs Only]({img_open})\n"  # Streamlit uses st.image separately
    else:
        images_section += "**📍 Solution Visualization:** Images available in UI/Solution_UI folder.\n"

    # Professional explanation can be appended externally
    return metrics + images_section
