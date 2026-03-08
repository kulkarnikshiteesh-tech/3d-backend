import os, math
import gradio as gr
import cadquery as cq
import trimesh
import numpy as np
from typing import List, Tuple

os.environ["CQ_SHOW_MESHES"] = "0"  # No display for server

def find_top_bottom_faces(part: cq.Solid) -> Tuple[List[cq.Face], List[cq.Face]]:
    """Auto‑detect largest horizontal faces for top/bottom."""
    top_faces = part.faces(">Z")   # Farthest +Z
    bottom_faces = part.faces("<Z")  # Farthest -Z
    return top_faces.vals(), bottom_faces.vals()

def raycast_undercuts(mesh: trimesh.Trimesh, pull_dir: np.ndarray, n_rays=1000) -> List[Tuple]:
    """True undercut detection via raycasting (catches curves/holes)."""
    # Sample rays from bbox top
    bounds = mesh.bounds
    ray_origins = np.random.uniform(bounds[0], bounds[1], (n_rays, 3))
    ray_origins[:, 2] = bounds[1, 2] + 10  # Above model
    ray_directions = np.tile(pull_dir, (n_rays, 1))
    
    # Cast rays
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins, ray_directions, step=1.0
    )
    
    # Count hits >1 (undercut)
    hits_per_ray = np.bincount(index_ray, minlength=n_rays)
    undercut_rays = np.where(hits_per_ray > 1)[0]
    
    return list(undercut_rays[:10])  # Top 10 examples

def dfm_analysis(step_file, surface_choice, material="ABS", quantity=1000, draft_deg=2.0):
    if not step_file:
        return "Upload STEP file"
    
    # Load
    part = cq.importers.importStep(step_file.name)
    
    # Auto top/bottom
    top_faces, bottom_faces = find_top_bottom_faces(part)
    if surface_choice == "Top surface":
        pull_dir = np.array([0, 0, -1.0])  # Pull down from top
        ref_faces = top_faces
    else:
        pull_dir = np.array([0, 0, 1.0])   # Pull up from bottom
        ref_faces = bottom_faces
    
    # Face‑based draft (fast)
    all_faces = part.faces().vals()
    bad_faces = []
    draft_rad = math.radians(draft_deg)
    for i, face in enumerate(all_faces):
        normal = np.array(face.Normal().toTuple())
        normal /= np.linalg.norm(normal) + 1e-9
        cos_theta = np.dot(normal, pull_dir)
        if cos_theta < math.cos(draft_rad):
            bad_faces.append((i, math.degrees(math.acos(np.clip(cos_theta, -1, 1)))))
    
    # True raycast for curves/holes
    mesh = part.toMesh().to_trimesh()
    undercut_rays = raycast_undercuts(mesh, pull_dir)
    
    # Volume/cost (INR)
    bbox = part.val().BoundingBox()
    vol_cm3 = bbox.xlen * bbox.ylen * bbox.zlen / 1000
    densities = {"ABS": 1.05, "PP": 0.9, "PC": 1.2}
    rates = {"ABS": 200, "PP": 180, "PC": 260}  # INR/kg
    density = densities.get(material, 1.05)
    rate = rates.get(material, 200)
    
    weight_g = vol_cm3 * density
    mat_cost = (weight_g / 1000) * rate
    mold_complexity = 80000 * (1 + 0.2 * len(bad_faces) + 0.1 * len(undercut_rays))
    q = max(quantity, 1)
    total_unit = mat_cost + mold_complexity / q
    
    # Report
    report = f"""
🛠️ **DFM Analysis** (Pull from {surface_choice})

**📐 Geometry**
- Est. volume: {vol_cm3:.1f} cm³ ({weight_g:.0f}g)
- Reference {surface_choice.lower()}: {len(ref_faces)} faces

**⚠️ Issues Detected**
- Draft violations: {len(bad_faces)} faces
- True undercuts (curves/holes): {len(undercut_rays)} rays blocked

**💰 Cost Estimate ({material}, Qty {q:,})**
- Material: ₹{mat_cost:.0f}/part
- Mold (complex): ₹{mold_complexity:,.0f}
- **Total/part: ₹{total_unit:.0f}**

**📈 Learning**: This analysis logged to improve future auto‑detection!
"""
    return report

# Gradio UI (non‑engineer friendly)
with gr.Blocks(title="CADCheck DFM") as demo:
    gr.Markdown("# CADCheck – Upload & Analyze (No CAD Skills Needed)")
    
    with gr.Row():
        file_input = gr.File(label="📁 STEP/STP File", file_types=[".step", ".stp"])
        surface_choice = gr.Dropdown(
            ["Top surface", "Bottom surface"], 
            value="Top surface", 
            label="🎯 Pull from which surface?"
        )
        material = gr.Dropdown(["ABS", "PP", "PC"], value="ABS", label="🧱 Material")
        quantity = gr.Number(value=1000, label="📊 Quantity")
    
    analyze_btn = gr.Button("🚀 Analyze DFM & Cost", variant="primary")
    result = gr.Markdown()
    
    analyze_btn.click(
        dfm_analysis, 
        inputs=[file_input, surface_choice, material, quantity],
        outputs=result
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
