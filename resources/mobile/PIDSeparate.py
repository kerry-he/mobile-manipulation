finished_ang = False

def step_pid(r, r_cam, Tep):

    wTb = r._base.A

    bTbp = np.linalg.inv(wTb) @ Tep

    # Spatial error
    bt = np.sum(np.abs(bTbp[:3, -1]))

    vb_lin = (np.linalg.norm(bTbp[:2, -1]) - 0.9) * 5
    vb_ang = np.arctan2(bTbp[1, -1], bTbp[0, -1]) * 50 * min(1.0, vb_lin/8.0)

    vb_lin = max(min(vb_lin, r_cam.qdlim[1]), -r_cam.qdlim[1])
    vb_ang = max(min(vb_ang, r_cam.qdlim[0]), -r_cam.qdlim[0])  
    
    global finished_ang
    if not finished_ang:
        vb_lin = 0
        finished_ang = abs(vb_ang / 50) < 0.01

    if bt < 1.75:
        arrived = True
        vb_lin = 0.0
        vb_ang = 0.0
    else:
        arrived = False




    # Simple camera PID
    wTc = r_cam.fkine(r_cam.q, fast=True)
    cTep = np.linalg.inv(wTc) @ Tep

    # Spatial error
    head_rotation, head_angle, _ = transform_between_vectors(np.array([1, 0, 0]), cTep[:3, 3])

    yaw = max(min(head_rotation.rpy()[2] * 10, r_cam.qdlim[3]), -r_cam.qdlim[3])
    pitch = max(min(head_rotation.rpy()[1] * 10, r_cam.qdlim[4]), -r_cam.qdlim[4])


    # Solve for the joint velocities dq
    qd = np.array([vb_ang, vb_lin, 0., 0., 0., 0., 0., 0., 0., 0.])
    qd_cam = np.array([vb_ang, vb_lin, 0., yaw, pitch])

    if bt > 0.5:
        qd *= 0.7 / bt
        qd_cam *= 0.7 / bt
    else:
        qd *= 1.4
        qd_cam *= 1.4

    print(vb_ang, vb_lin)
    print(bt)

    return arrived, qd, qd_cam        