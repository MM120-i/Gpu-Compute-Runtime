use runtime::dispatcher::{Dispatcher, WorkgroupCount};

#[test]
fn workgroup_count_1d_exact() {
    let wg: WorkgroupCount = Dispatcher::workgroup_count_1d(64, 64);
    assert_eq!(wg.x, 1);
    assert_eq!(wg.y, 1);
    assert_eq!(wg.z, 1);
}

#[test]
fn workgroup_count_1d_round_up() {
    let wg: WorkgroupCount = Dispatcher::workgroup_count_1d(65, 64);
    assert_eq!(wg.x, 2);
}

#[test]
fn workgroup_count_1d_one_element() {
    let wg: WorkgroupCount = Dispatcher::workgroup_count_1d(1, 64);
    assert_eq!(wg.x, 1);
}

#[test]
fn workgroup_count_1d_zero_elements() {
    let wg: WorkgroupCount = Dispatcher::workgroup_count_1d(0, 64);
    assert_eq!(wg.x, 0);
}

#[test]
fn workgroup_count_1d_large_input() {
    let wg: WorkgroupCount = Dispatcher::workgroup_count_1d(1_000_000, 256);
    assert_eq!(wg.x, 3907);
}

#[test]
fn workgroup_construct() {
    let wg: WorkgroupCount = WorkgroupCount { x: 8, y: 4, z: 2 };
    assert_eq!(wg.x, 8);
    assert_eq!(wg.y, 4);
    assert_eq!(wg.z, 2);
}
