import sys
import numpy as np

def parse_dh_param_file(dh_config_file):
    assert(dh_config_file is not None)
    f_line_contents = None
    with open(dh_config_file, "r") as f:
        f_line_contents = f.readlines()

    assert(f.closed)
    assert(f_line_contents is not None)
    # maybe not the most efficient/clean/etc. way to do this, but should only have to be done once so NBD
    dh_params = np.asarray([line.rstrip().split(',') for line in f_line_contents[1:]])
    dh_params = dh_params.astype(float)
    return dh_params


### TODO: parse a pox parameter file
def parse_pox_param_file(pox_config_file):
    """!
    @brief      Parses a POX parameter file.

    @param      pox_config_file  The pox configuration file path

    @return     M matrix (4x4) and S list (6xN) as numpy arrays
    """
    assert(pox_config_file is not None)
    
    m_rows = []
    s_rows = []
    current_section = None

    with open(pox_config_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Identify which section we are in
            if "# M matrix" in line:
                current_section = "M"
                continue
            elif "# screw vectors" in line:
                current_section = "S"
                continue
            
            # Process the data rows
            # Handle both comma-separated and space-separated values
            vals = line.replace(',', ' ').split()
            if not vals:
                continue
            
            if current_section == "M":
                m_rows.append([float(x) for x in vals])
            elif current_section == "S":
                s_rows.append([float(x) for x in vals])

    # Convert to numpy arrays
    m_mat = np.array(m_rows)
    s_lst = np.array(s_rows)

    return m_mat, s_lst