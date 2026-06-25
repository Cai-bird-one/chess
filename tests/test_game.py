from xiangzero.game import Move, NATURAL_DRAW_PLIES, XiangqiGame


def test_initial_position_has_legal_moves():
    game = XiangqiGame()
    moves = game.legal_moves()
    assert moves
    assert all(game.apply(move) for move in moves)


def test_move_roundtrip_action():
    move = Move(89, 80)
    assert Move.from_action(move.action) == move


def test_kings_may_not_face_after_move():
    game = XiangqiGame("....k...." + "." * 72 + "....K....")
    assert game.is_in_check(1)


def test_flying_general_is_not_generated_as_a_legal_move():
    game = XiangqiGame("....k...." + "." * 72 + "....K....")
    assert "e0e9" not in {move.ucci() for move in game.legal_moves()}


def test_threefold_repetition_is_draw():
    game = XiangqiGame()
    game = XiangqiGame(game.board, game.side, history=(game.position_key(),) * 3)
    result = game.result()
    assert result.terminal
    assert result.value == 0.0
    assert result.reason == "threefold_repetition"


def test_natural_draw_after_sixty_rounds_without_capture():
    game = XiangqiGame(halfmove_clock=NATURAL_DRAW_PLIES)
    result = game.result()
    assert result.terminal
    assert result.value == 0.0
    assert result.reason == "natural_draw"


def test_capture_resets_halfmove_clock():
    cells = ["."] * 90
    cells[4] = "k"
    cells[84] = "r"
    cells[85] = "K"
    board = "".join(cells)
    game = XiangqiGame(board, halfmove_clock=20)
    child = game.apply(Move(85, 84))
    assert child.halfmove_clock == 0
