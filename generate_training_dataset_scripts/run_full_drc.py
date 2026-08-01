import subprocess
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import SKY130_DRC_SCRIPT, find_klayout_executable


def run_m1_2_drc_python(input_gds, output_rdb):
    """Portable m1.2-only fallback using the KLayout Python package."""
    try:
        import klayout.db as db
        import klayout.rdb as rdb
    except ImportError:
        print("[!] Neither the KLayout executable nor Python package is available.")
        return False

    layout = db.Layout()
    layout.read(str(input_gds))
    top_cell = layout.top_cell()
    metal_index = layout.find_layer(68, 20)
    if metal_index is None:
        print("[!] Metal 1 layer 68/20 was not found.")
        return False

    metal = db.Region(top_cell.begin_shapes_rec(metal_index))
    metal.merge()
    minimum_spacing_dbu = round(0.140 / layout.dbu)
    violations = metal.space_check(minimum_spacing_dbu)

    report = rdb.ReportDatabase("DRC_Database")
    category = report.create_category(None, "m1.2")
    cell = report.create_cell(top_cell.name)
    for edge_pair in violations.each():
        item = report.create_item(cell.rdb_id(), category.rdb_id())
        item.add_value(
            db.DEdgePair(
                edge_pair.first.to_dtype(layout.dbu),
                edge_pair.second.to_dtype(layout.dbu),
            )
        )

    Path(output_rdb).parent.mkdir(parents=True, exist_ok=True)
    report.save(str(output_rdb))
    print(f"[*] Python m1.2 DRC found {violations.count()} edge pair(s).")
    print(f"[*] Report saved to: {output_rdb}")
    return True


def run_full_drc(input_gds, output_rdb):
    """
    Runs KLayout in batch mode (no GUI) to execute a DRC script.
    It passes the input GDS and the desired output RDB file as variables.
    """
    drc_script = SKY130_DRC_SCRIPT
    klayout_cmd = find_klayout_executable()

    if not drc_script.exists():
        print(f"Error: DRC script '{drc_script}' not found!")
        print("Please ensure the sky130 DRC rule deck is in the project folder.")
        return False

    abs_input = str(Path(input_gds).resolve())
    abs_output = str(Path(output_rdb).resolve())

    print(f"[*] Starting KLayout DRC Engine in batch mode...")
    print(f"[*] Using KLayout executable: {klayout_cmd}")

    executable = shutil.which(klayout_cmd)
    if executable is None and Path(klayout_cmd).is_file():
        executable = klayout_cmd
    if executable is None:
        print("[*] KLayout CLI not found; using the Python m1.2-only fallback.")
        return run_m1_2_drc_python(abs_input, abs_output)

    command = [
        executable,
        "-b",
        "-r", str(drc_script),
        "-rd", f"input={abs_input}",
        "-rd", f"report={abs_output}",
    ]

    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"[*] DRC completed successfully. Report saved to: {output_rdb}")
        return True

    except subprocess.CalledProcessError as e:
        print("!!! DRC Execution Failed !!!")
        print("Error output from KLayout:")
        print(e.stderr)
        return False
    except FileNotFoundError as e:
        print(f"[!] KLayout executable could not be started: {e}")
        return run_m1_2_drc_python(abs_input, abs_output)


if __name__ == "__main__":
    from project_paths import injected_m1_gds, drc_report_path

    layout_name = "tt_um_yen"
    run_full_drc(injected_m1_gds(layout_name), drc_report_path(layout_name))
