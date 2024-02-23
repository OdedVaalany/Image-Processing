from ex1 import main


def test_one():
    ans = main("../ex1/vids/video1_category1.mp4", "category1")
    assert ans[0] == 99
    assert ans[1] == 100


def test_two():
    ans = main("../ex1/vids/video2_category1.mp4", "category1")
    assert ans[0] == 149
    assert ans[1] == 150


def test_three():
    ans = main("../ex1/vids/video3_category2.mp4", "category2")
    assert ans[0] == 174
    assert ans[1] == 175


def test_four():
    ans = main("../ex1/vids/video4_category2.mp4", "category2")
    assert ans[0] == 74
    assert ans[1] == 75


def test_five():
    ans = main("../ex1/vids/video5_category1.mp4", "category1")
    assert ans[0] == 99
    assert ans[1] == 100


def test_six():
    ans = main("../ex1/vids/video6_category1.mp4", "category1")
    assert ans[0] == 149
    assert ans[1] == 150


def test_seven():
    ans = main("../ex1/vids/video7_category2.mp4", "category2")
    assert ans[0] == 174
    assert ans[1] == 175


def test_eight():
    ans = main("../ex1/vids/video8_category2.mp4", "category2")
    assert ans[0] == 74
    assert ans[1] == 75
