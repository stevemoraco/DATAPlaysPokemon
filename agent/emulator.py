import io
import logging
import pickle
from collections import deque
import heapq
import os

from agent.memory_reader import PokemonRedReader, StatusCondition
from PIL import Image
from pyboy import PyBoy

logger = logging.getLogger(__name__)


class Emulator:
    def __init__(self, rom_path, headless=True, sound=False):
        self.rom_path = rom_path  # Store the ROM path
        self.headless = headless  # Store headless state
        self.sound = sound  # Store sound state
        try:
            # First try with CGB mode
            if headless:
                # Pass sound flag even in headless mode
                self.pyboy = PyBoy(rom_path, window="null", sound=sound, cgb=True)
            else:
                self.pyboy = PyBoy(rom_path, sound=sound, cgb=True)
        except Exception:
            logger.info("Failed to initialize in CGB mode, falling back to GB mode")
            # Fallback to GB mode
            if headless:
                self.pyboy = PyBoy(rom_path, window="null", sound=sound, cgb=False)
            else:
                self.pyboy = PyBoy(rom_path, sound=sound, cgb=False)

    def tick(self, frames):
        """Advance the emulator by the specified number of frames."""
        for _ in range(frames):
            self.pyboy.tick()

    def initialize(self):
        """Initialize the emulator."""
        # Run the emulator for a short time to make sure it's ready
        self.pyboy.set_emulation_speed(0)
        for _ in range(60):
            self.tick(60)
        self.pyboy.set_emulation_speed(1)

    def get_screenshot(self):
        """Get the current screenshot."""
        return Image.fromarray(self.pyboy.screen.ndarray)

    def get_screenshot_with_overlay(self, alpha=128):
        """
        Get the current screenshot with a tile overlay showing walkable/unwalkable areas.
        
        Args:
            alpha (int): Transparency value for the overlay (0-255)
            
        Returns:
            PIL.Image: Screenshot with tile overlay
        """
        from tile_visualizer import overlay_on_screenshot
        screenshot = self.get_screenshot()
        collision_map = self.get_collision_map()
        return overlay_on_screenshot(screenshot, collision_map, alpha)

    def load_state(self, state_filename):
        """
        Load a PyBoy save state file into the emulator.
        
        Args:
            state_filename: Path to the PyBoy .state file
        """
        try:
            with open(state_filename, 'rb') as f:
                state_data = f.read()
                state_io = io.BytesIO(state_data)
                self.pyboy.load_state(state_io)
        except Exception as e:
            # If direct loading fails, try with pickle
            try:
                with open(state_filename, 'rb') as f:
                    state_data = pickle.load(f)
                    if "pyboy_state" in state_data:
                        pyboy_state_io = io.BytesIO(state_data["pyboy_state"])
                        self.pyboy.load_state(pyboy_state_io)
                    else:
                        raise ValueError("Invalid save state format")
            except Exception as e2:
                logger.error(f"Failed to load save state: {e2}")
                raise
    
    def save_state(self, state_filename):
        """
        Save a PyBoy save state to a file.
        Args:
            state_filename: Path to write the save state (.state file)
        """
        try:
            # Ensure target directory exists
            os.makedirs(os.path.dirname(state_filename), exist_ok=True)
            with open(state_filename, 'wb') as f:
                # PyBoy supports saving directly to a file-like
                self.pyboy.save_state(f)
        except Exception as e:
            logger.error(f"Failed to save state to {state_filename}: {e}")
            raise

    def press_buttons(self, buttons, wait=True):
        """Press a sequence of buttons on the Game Boy.
        
        Args:
            buttons (list[str]): List of buttons to press in sequence
            wait (bool): Whether to wait after each button press
            
        Returns:
            str: Result of the button presses
        """
        results = []

        for button in buttons:
            if button not in ["a", "b", "start", "select", "up", "down", "left", "right"]:
                results.append(f"Invalid button: {button}")
                continue
            # Execute the valid press
            self.pyboy.button_press(button)
            self.tick(10)   # Press briefly
            self.pyboy.button_release(button)

            # Wait after press if requested
            self.tick(120 if wait else 10)

            results.append(f"Pressed {button}")

        return "\n".join(results)

    def get_coordinates(self):
        """
        Return player's position as (row, column) to match the 9×10 grid and
        path‑finding helpers.  For legacy uses that expect the original
        (x, y) order, call ``get_coordinates_xy``.
        Returns:
            tuple[int, int]: (row, column) coordinates
        """
        reader = PokemonRedReader(self.pyboy.memory)
        # read_coordinates returns (row, col)
        return reader.read_coordinates()

    # ------------------------------------------------------------------
    # Backwards‑compat shim – returns (x, y)
    # ------------------------------------------------------------------

    def get_coordinates_xy(self):
        """Return coordinates in the original Game Boy order (x, y)."""
        row, col = self.get_coordinates()
        return (col, row)

    def get_active_dialog(self):
        """
        Returns the active dialog text from game memory.
        Returns:
            str: Dialog text
        """
        reader = PokemonRedReader(self.pyboy.memory)
        dialog = reader.read_dialog()
        if dialog:
            return dialog
        return None

    def get_location(self):
        """
        Returns the player's current location name from game memory.
        Returns:
            str: Location name
        """
        reader = PokemonRedReader(self.pyboy.memory)
        return reader.read_location()

    def _get_direction(self, array):
        """Determine the player's facing direction from the sprite pattern."""
        # Look through the array for any 2x2 grid containing numbers 0-3
        rows, cols = array.shape

        for i in range(rows - 1):
            for j in range(cols - 1):
                # Extract 2x2 grid
                grid = array[i : i + 2, j : j + 2].flatten()

                # Check for each direction pattern
                if list(grid) == [0, 1, 2, 3]:
                    return "down"
                elif list(grid) == [4, 5, 6, 7]:
                    return "up"
                elif list(grid) == [9, 8, 11, 10]:
                    return "right"
                elif list(grid) == [8, 9, 10, 11]:
                    return "left"

        return "no direction found"

    def _get_player_center(self, array):
        """Locate the 2×2 sprite block that represents the player and return
        the centre (row, col) within the 18×20 screen grid.  Falls back to
        (9,8) if the pattern is not found.
        """
        rows, cols = array.shape

        patterns = [
            ([0, 1, 2, 3], "down"),   # facing down
            ([4, 5, 6, 7], "up"),     # facing up
            ([9, 8, 11, 10], "right"),
            ([8, 9, 10, 11], "left"),
        ]

        for i in range(rows - 1):
            for j in range(cols - 1):
                block = array[i : i + 2, j : j + 2].flatten().tolist()
                for pattern, _ in patterns:
                    if block == pattern:
                        return i + 1, j + 1  # centre of 2×2 block
        # Fallback to assumed centre of screen
        return 9, 8

    def _downsample_array(self, arr):
        """Downsample an 18x20 array to 9x10 by averaging 2x2 blocks."""
        # Ensure input array is 18x20
        if arr.shape != (18, 20):
            raise ValueError("Input array must be 18x20")

        # Reshape to group 2x2 blocks and take mean
        return arr.reshape(9, 2, 10, 2).mean(axis=(1, 3))

    def get_collision_map(self):
        """
        Creates a simple ASCII map showing player position, direction, terrain and sprites.
        Takes into account tile pair collisions for more accurate walkability.
        Returns:
            str: A string representation of the ASCII map with legend
        """
        # Get the terrain and movement data
        full_map = self.pyboy.game_area()
        collision_map = self.pyboy.game_area_collision()
        downsampled_terrain = self._downsample_array(collision_map)

        # Get sprite locations
        sprite_locations = self.get_sprites()

        # Get character direction from the full map
        direction = self._get_direction(full_map)
        if direction == "no direction found":
            return None

        # Prepare collision lookup
        reader = PokemonRedReader(self.pyboy.memory)
        tileset = reader.read_tileset()
        full_tilemap = self.pyboy.game_wrapper._get_screen_background_tilemap()

        # Numeric codes: 0=walkable, 1=wall, 2=sprite, 3=player up, 4=player down, 5=player left, 6=player right
        dir_codes = {"up": 3, "down": 4, "left": 5, "right": 6}
        player_code = dir_codes.get(direction, 3)

        # Build numeric grid
        grid = []
        for i in range(9):
            row = []
            for j in range(10):
                # Player at center
                if i == 4 and j == 4:
                    row.append(player_code)
                # Sprite positions
                elif (j, i) in sprite_locations:
                    row.append(2)
                else:
                    # Base terrain check
                    walkable = False
                    if downsampled_terrain[i][j] != 0:
                        current_tile = full_tilemap[i * 2 + 1][j * 2]
                        player_tile = full_tilemap[9][8]
                        if self._can_move_between_tiles(player_tile, current_tile, tileset):
                            walkable = True
                    # Append code
                    row.append(0 if walkable else 1)
            grid.append(row)

        # Prepare output lines
        lines = []
        for row in grid:
            lines.append(" ".join(str(x) for x in row))

        # Legend for numeric codes
        lines.extend([
            "",
            "Legend:",
            "0 - walkable path",
            "1 - wall / obstacle / unwalkable",
            "2 - sprite (NPC)",
            "3 - player (facing up)",
            "4 - player (facing down)",
            "5 - player (facing left)",
            "6 - player (facing right)",
        ])
        return "\n".join(lines)

    def get_valid_moves(self):
        """Return list of valid cardinal directions for the player this frame.

        Uses the full 18×20 collision grid so single‑tile warps/doors are not
        lost in down‑sampling.  Additionally, certain tile IDs are treated as
        walkable even if the collision byte is 0 (warp/door tiles in Pokémon
        Red).
        """

        collision = self.pyboy.game_area_collision()  # 18×20 ints (0/1)
        # The background tilemap (same resolution) lets us identify warps
        full_map = self.pyboy.game_wrapper._get_screen_background_tilemap()

        # Known warp/door tile indices (inside houses, building exits, etc.)
        WARP_TILE_IDS = {
            # stair warp tiles
            0x0A, 0x0B,
            # interior door top/bottom
            0x4E, 0x4F,
            # exterior single‑door top/bottom variants
            0x50, 0x51, 0x52, 0x53,
            # house / lab door variants
            0x5E, 0x5F,
            0x6E, 0x6F,
            0x70, 0x71, 0x72, 0x73,
        }

        # Helper to decide if the tile at (r,c) can be entered
        def is_walkable(r: int, c: int) -> bool:
            if not (0 <= r < 18 and 0 <= c < 20):
                return False
            if collision[r][c] != 0:
                return True
            # collision == 0  => normally a wall; allow if warp tile id
            return full_map[r][c] in WARP_TILE_IDS

        # Locate player sprite dynamically (works after map scroll)
        pr, pc = self._get_player_center(full_map)
        directions = {
            "up": (pr - 1, pc),
            "down": (pr + 1, pc),
            "left": (pr, pc - 1),
            "right": (pr, pc + 1),
        }

        valid = [d for d, (r, c) in directions.items() if is_walkable(r, c)]

        # If standing on a warp tile, always allow the direction that leads off‑screen
        if full_map[pr][pc] in WARP_TILE_IDS:
            # Determine facing direction to exit (depends on warp orientation)
            # crude heuristic: if pr < 9 then up exits, if pr > 9 down exits
            if pr <= 8 and "up" not in valid:
                valid.append("up")
            if pr >= 9 and "down" not in valid:
                valid.append("down")
        return valid

    def _can_move_between_tiles(self, tile1: int, tile2: int, tileset: str) -> bool:
        """
        Check if movement between two tiles is allowed based on tile pair collision data.

        Args:
            tile1: The tile being moved from
            tile2: The tile being moved to
            tileset: The current tileset name

        Returns:
            bool: True if movement is allowed, False if blocked
        """
        # Tile pair collision data
        TILE_PAIR_COLLISIONS_LAND = [
            ("CAVERN", 288, 261),
            ("CAVERN", 321, 261),
            ("FOREST", 304, 302),
            ("CAVERN", 298, 261),
            ("CAVERN", 261, 289),
            ("FOREST", 338, 302),
            ("FOREST", 341, 302),
            ("FOREST", 342, 302),
            ("FOREST", 288, 302),
            ("FOREST", 350, 302),
            ("FOREST", 351, 302),
        ]

        TILE_PAIR_COLLISIONS_WATER = [
            ("FOREST", 276, 302),
            ("FOREST", 328, 302),
            ("CAVERN", 276, 261),
        ]

        # Check both land and water collisions
        for ts, t1, t2 in TILE_PAIR_COLLISIONS_LAND + TILE_PAIR_COLLISIONS_WATER:
            if ts == tileset:
                # Check both directions since collisions are bidirectional
                if (tile1 == t1 and tile2 == t2) or (tile1 == t2 and tile2 == t1):
                    return False

        return True

    def get_sprites(self, debug=False):
        """
        Get the location of all of the sprites on the screen.
        returns set of coordinates that are (column, row)
        """
        # Group sprites by their exact Y coordinate
        sprites_by_y = {}

        for i in range(40):
            sp = self.pyboy.get_sprite(i)
            if sp.on_screen:
                x = int(sp.x / 160 * 10)
                y = int(sp.y / 144 * 9)
                orig_y = sp.y

                if orig_y not in sprites_by_y:
                    sprites_by_y[orig_y] = []
                sprites_by_y[orig_y].append((x, y, i))

        # Sort Y coordinates
        y_positions = sorted(sprites_by_y.keys())
        bottom_sprite_tiles = set()

        if debug:
            print("\nSprites grouped by original Y:")
            for orig_y in y_positions:
                sprites = sprites_by_y[orig_y]
                print(f"Y={orig_y}:")
                for x, grid_y, i in sprites:
                    print(f"  Sprite {i}: x={x}, grid_y={grid_y}")

        SPRITE_HEIGHT = 8

        # First, group sprites by X coordinate for each Y level
        for i in range(len(y_positions) - 1):
            y1 = y_positions[i]
            y2 = y_positions[i + 1]

            if y2 - y1 == SPRITE_HEIGHT:
                # Group sprites by X coordinate at each Y level
                sprites_at_y1 = {s[0]: s for s in sprites_by_y[y1]}  # x -> sprite info
                sprites_at_y2 = {s[0]: s for s in sprites_by_y[y2]}

                # Only match sprites that share the same X coordinate
                for x in sprites_at_y2:
                    if x in sprites_at_y1:  # If there's a matching top sprite at this X
                        bottom_sprite = sprites_at_y2[x]
                        bottom_sprite_tiles.add((x, bottom_sprite[1]))
                        if debug:
                            print(f"\nMatched sprites at x={x}, Y1={y1}, Y2={y2}")

        return bottom_sprite_tiles

    # ------------------------------------------------------------------
    # Warp / Door detection helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Door tile ID sets – top halves, bottom halves, and single‑tile stairs
    # ------------------------------------------------------------------

    # Actual single‑tile warp stairs (bottom step). Top‑half graphics 0x1E/0x1F
    # are **not** warps and must be excluded or they create false doors.
    _STAIR_TILES = {0x0A, 0x0B, 0x1A, 0x1B, 0x1C, 0x1D}

    # ------------------------------------------------------------------
    # Door tile IDs
    # ------------------------------------------------------------------

    # Older logic tried to infer doors by matching a TOP‑tile directly above a
    # BOTTOM‑tile.  In practice the full tilemap scrolls, NPC sprites overlap
    # the graphics, and many legitimate warp tiles (e.g. cave exits) are
    # single‑tile, making that approach brittle.  Instead we maintain a single
    # flat set that lists **only** the tile IDs that the game engine uses as
    # the *walk‑into* warp tile – the bottom half of doors and the staircase
    # step.  This greatly simplifies detection and eliminates duplicate /
    # mismatched pairs.

    _DOOR_WARP_IDS = {
        # Warp tile list – add 0x1B (exterior house door bottom)

        0x4F,   # interior door bottom
        0x34,   # staircase bottom
        0x1B,   # exterior house/lab door bottom
    }

    # Re‑use _STAIR_TILES so stairs are always included even if list drifts
    _DOOR_TILE_IDS = _STAIR_TILES | _DOOR_WARP_IDS

    # Manual mapping from certain interior map names to their exterior location.
    # This is deliberately minimal – we only include early‑game interiors for now.
    _INTERIOR_DEST_OVERRIDES = {
        "PLAYERS HOUSE 1F": "Pallet Town",
        # For staircases inside the house, upstairs leads to 1F, not outdoors
        "PLAYERS HOUSE 2F": "Players House 1F",
        "OAKS LAB": "Pallet Town",
        "RIVALS HOUSE": "Pallet Town",
    }

    def _infer_door_destination(self, current_location: str) -> str | None:
        """Best‑effort guess of the exterior destination for a door.

        The approach is heuristic – for certain known interiors we return a
        hard‑coded town/city.  For generic buildings whose name starts with a
        town/city (e.g. "VIRIDIAN POKECENTER") we strip the building type and
        append the proper suffix ("City"/"Town") when possible.
        """

        # Direct overrides first
        if current_location in self._INTERIOR_DEST_OVERRIDES:
            return self._INTERIOR_DEST_OVERRIDES[current_location]

        tokens = current_location.split()
        if not tokens:
            return None

        first = tokens[0].capitalize()

        # Known town/city keywords to help choose suffix
        towns = {
            "Pallet": "Town",
            "Lavender": "Town",
            "Viridian": "City",
            "Pewter": "City",
            "Cerulean": "City",
            "Vermilion": "City",
            "Celadon": "City",
            "Fuchsia": "City",
            "Saffron": "City",
            "Cinnabar": "Island",
            "Indigo": "Plateau",
        }

        if first in towns:
            return f"{first} {towns[first]}"

        # If the location name already ends with City/Town/etc. don't modify
        if tokens[-1] in {"Town", "City", "Island", "Plateau", "Route"}:
            return current_location

        return None

    def _get_doors_info(self) -> list[tuple[str | None, tuple[int, int]]]:
        """Return a list of visible warps using the game's warp table.

        Each entry is ``(destination_name_or_None, (row, col))`` where
        ``row`` and ``col`` are the absolute map‑tile coordinates read
        directly from WRAM.  Because these come from ``wWarpEntries`` they do
        **not** depend on the camera and therefore never jitter.
        """

        # ------------------------------------------------------------------
        # 1. Read warp entries for this map from WRAM
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Fallback to viewport tile‑scan (stable & working)
        # ------------------------------------------------------------------
        # Use ONLY the 18×20 viewport that is currently visible on‑screen so
        # we never report off‑screen doors.  This makes the coordinate system
        # match exactly what the player sees and what the collision/overlay
        # map shows.
        full_map = self.pyboy.game_wrapper._get_screen_background_tilemap()

        doors: list[tuple[int, int]] = []  # store down‑sampled cell coords

        # --------------------------------------------------------------
        # Exact 2×2 pattern whitelist (UL, UR, LL, LR).  None = wildcard.
        # --------------------------------------------------------------
        PATTERNS = [
            # Exterior house / lab door
            (0x0B, 0x0C, 0x1B, 0x1C),
            # Player house staircase bottom
            (0x34, 0x1E, 0x34, 0x1F),
            # Interior single door (UR wildcard, lower row wildcard)
            (0x4F, 0x4E, None, None),
        ]

        def match_block(tl, tr, bl, br):
            for a, b, c_, d in PATTERNS:
                if (a is None or tl == a) and (b is None or tr == b) and (
                    c_ is None or bl == c_
                ) and (d is None or br == d):
                    return True
            return False

        # Screen viewport fixed 18×20; iterate by 2×2 blocks
        for base_r in range(0, 18, 2):
            for base_c in range(0, 20, 2):
                if base_r + 1 >= 18 or base_c + 1 >= 20:
                    continue

                tl = full_map[base_r][base_c] & 0xFF
                tr = full_map[base_r][base_c + 1] & 0xFF
                bl = full_map[base_r + 1][base_c] & 0xFF
                br = full_map[base_r + 1][base_c + 1] & 0xFF

                if not match_block(tl, tr, bl, br):
                    continue

                ds_r, ds_c = base_r // 2, base_c // 2
                doors.append((ds_r, ds_c, tl))

        # De‑duplicate by down‑sampled coordinates (2×2 => 1 block)
        # Log raw door positions with tile IDs for debugging
        if logger.isEnabledFor(logging.DEBUG):
            try:
                logger.debug(
                    "[DoorDetect] door tile list (row,col,tileHex): "
                    + str([(r, c, hex(full_map[r][c] & 0xFF)) for r, c in doors])
                )
            except Exception:
                pass

        # Log full 18×20 background tile hex grid for manual pattern work
        full_dump = [
            " ".join(hex(t & 0xFF)[2:].upper().zfill(2) for t in row)
            for row in full_map
        ]
        # Verbose full‑map dump can flood logs; keep it at DEBUG level.
        logger.debug("[DoorDetect] full 18x20 background tile IDs:\n" + "\n".join(full_dump))

        # Down‑sampled coordinate → tile_id for every warp tile we found.
        # Using only the warp tile list already removes door tops, so no extra
        # filtering is necessary.
        unique_coords: dict[tuple[int, int], int] = {}
        unique_coords: dict[tuple[int, int], int] = {}
        for ds_r, ds_c, tid in doors:
            unique_coords[(ds_r, ds_c)] = tid

        # Validate each down‑sampled 2×2 block: keep it *only* if it contains
        # the canonical stair/door warp tile **and** a matching graphic from
        # the same staircase pair.  For interior staircases that means one of
        # the top‑half graphics 0x1E/0x1F together with bottom warp 0x34.  This
        # removes stray 0x34 tiles that appear elsewhere in furniture.

        def has_stair_pattern(ds_r: int, ds_c: int) -> bool:
            base_r, base_c = ds_r * 2, ds_c * 2
            if base_r + 1 >= 18 or base_c + 1 >= 20:
                return False
            tiles = {
                full_map[base_r][base_c] & 0xFF,
                full_map[base_r][base_c + 1] & 0xFF,
                full_map[base_r + 1][base_c] & 0xFF,
                full_map[base_r + 1][base_c + 1] & 0xFF,
            }
            # Require warp tile and at least one stair‑top tile
            return 0x34 in tiles and bool(tiles & {0x1E, 0x1F})

        unique_coords = {
            (ds_r, ds_c): tid
            for (ds_r, ds_c), tid in unique_coords.items()
            if (
                # if warp tile is 0x34 we require full pattern; for other warp
                # ids we keep them as is (single‑tile cave exits, doors etc.)
                (tid != 0x34) or has_stair_pattern(ds_r, ds_c)
            )
        }

        # ---------- diagnostic logging ---------------------------------------------------
        # Log both the raw (row,col) positions and the hex tile IDs that survived the
        # filtering so false positives are easy to spot in the runtime logs.

        if doors:
            raw_with_hex = [(r, c, hex(tid)) for r, c, tid in doors]
            kept_with_hex = [
                (ds_r, ds_c, hex(tid)) for (ds_r, ds_c), tid in unique_coords.items()
            ]
            logger.debug(
                "[DoorDetect] found %d warp‑candidate tiles in %s | raw=%s kept=%s",
                len(doors),
                self.get_location(),
                raw_with_hex,
                kept_with_hex,
            )
        else:
            logger.debug("[DoorDetect] found 0 warp tiles in %s", self.get_location())

        # Diagnostic: dump down‑sampled 9×10 background tile IDs (bottom‑left of
        # each 2×2 block) so we can compare with collision map coordinates.
        try:
            ds_rows = []
            for ds_r in range(9):
                row_ids = []
                for ds_c in range(10):
                    tile_id = full_map[ds_r * 2 + 1][ds_c * 2] & 0xFF
                    row_ids.append(hex(tile_id)[2:].upper().zfill(2))
                ds_rows.append(" ".join(row_ids))
            logger.info("[DoorDetect] down‑sampled 9x10 tile IDs:\n" + "\n".join(ds_rows))
        except Exception:
            pass

        # Extra diagnostic: if none found, log tile IDs at player column across
        # the bottom six rows to help identify unknown door tiles.
        if not doors:
            try:
                full_map = self.pyboy.game_wrapper._get_screen_background_tilemap()
                player_r, player_c = self._get_player_center(full_map)
                sample = [full_map[r][player_c] for r in range(12, 18)]
                logger.info(
                    f"[DoorDetect] sampling column {player_c} rows 12‑17 tile IDs: {sample}"
                )

                # Dump the entire 18×20 tile id grid once (compact)
                grid_flat = [hex(t)[2:].upper().zfill(2) for row in full_map for t in row]
                rows_str = [" ".join(grid_flat[i * 20 : (i + 1) * 20]) for i in range(18)]
                logger.info("[DoorDetect] full 18x20 background tilemap:\n" + "\n".join(rows_str))
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Convert to on‑screen 9×10 coordinates; include only doors visible in
        # the current viewport so the numbers stay meaningful for navigate_to.
        # ------------------------------------------------------------------

        # We now convert each unique down‑sampled cell to world‑tile
        # coordinates using the player’s position as the origin.

        try:
            player_row, player_col = self.get_coordinates()
        except Exception:
            player_row = player_col = 0

        location_name = self.get_location() or ""
        dest_guess = self._infer_door_destination(location_name)

        visible_doors: list[tuple[str | None, tuple[int, int]]] = []

        for (ds_r, ds_c), tid in unique_coords.items():
            delta_cells_r = ds_r - 4  # relative to player cell (4,4)
            delta_cells_c = ds_c - 4

            # In this emulator build each 9×10 cell corresponds to **one**
            # world tile (not two) because the player position is restricted
            # to whole‑tile increments that line up with the down‑sampled
            # grid.  Therefore apply the delta in cells directly.
            world_r = player_row + delta_cells_r
            world_c = player_col + delta_cells_c

            # Fine‑grained destination override for staircase tiles inside
            # the player’s house so the prompt doesn’t claim they lead to
            # Pallet Town.
            # If this warp is a staircase (tile 0x34) we cannot reliably infer
            # its destination from the location name heuristic, so omit the
            # label rather than risk a misleading "Pallet Town" message.
            dest_final = None if tid == 0x34 else dest_guess
            visible_doors.append((dest_final, (world_r, world_c)))

        # ------------------------------------------------------------------
        # Extra diagnostic: dump the exact 2×2 blocks that generated each
        # down‑sampled cell we report as a door so it is easy to curate the
        # warp‑tile list.
        # ------------------------------------------------------------------

        if visible_doors and logger.isEnabledFor(logging.DEBUG):
            blocks_info: list[str] = []
            # Skip detailed 2×2 dump in world‑coord mode – not easily mapped.
            pass

        logger.debug(f"[DoorDetect] visible_doors={visible_doors}")
        return visible_doors

    # ------------------------------------------------------------------
    # Diagnostics helpers for SimpleAgent logging
    # ------------------------------------------------------------------

    def _screen_origin(self) -> tuple[int, int]:
        """Return (cam_row, cam_col) world‑tile coordinates of viewport top‑left."""
        try:
            player_row, player_col = self.get_coordinates()
        except Exception:
            return (0, 0)
        return (player_row - 9, player_col - 8)

    def tile_hex_at(self, world_row: int, world_col: int) -> str | None:
        """Return background tile hex at given world coords if visible."""
        cam_row, cam_col = self._screen_origin()
        r = world_row - cam_row
        c = world_col - cam_col
        if 0 <= r < 18 and 0 <= c < 20:
            tile = self.pyboy.game_wrapper._get_screen_background_tilemap()[r][c] & 0xFF
            return hex(tile)[2:].upper().zfill(2)
        return None

    def block_hex_at(self, world_row: int, world_col: int) -> list[str]:
        """Return list of 4 hex tile IDs of the 2×2 block containing world tile."""
        cam_row, cam_col = self._screen_origin()
        r = world_row - cam_row
        c = world_col - cam_col
        if not (0 <= r < 18 and 0 <= c < 20):
            return []
        base_r = (r // 2) * 2
        base_c = (c // 2) * 2
        full_map = self.pyboy.game_wrapper._get_screen_background_tilemap()
        tiles = [
            full_map[base_r][base_c] & 0xFF,
            full_map[base_r][base_c + 1] & 0xFF if base_c + 1 < 20 else 0,
            full_map[base_r + 1][base_c] & 0xFF if base_r + 1 < 18 else 0,
            full_map[base_r + 1][base_c + 1] & 0xFF if base_r + 1 < 18 and base_c + 1 < 20 else 0,
        ]
        return [hex(t)[2:].upper().zfill(2) for t in tiles]

    def find_path(self, target_row: int, target_col: int) -> tuple[str, list[str]]:
        """
        Finds the most efficient path from the player's current position (4,4) to the target position.
        If the target is unreachable, finds path to nearest accessible spot.
        Allows ending on a wall tile if that's the target.
        Takes into account terrain, sprite collisions, and tile pair collisions.

        Args:
            target_row: Row index in the 9x10 downsampled map (0-8)
            target_col: Column index in the 9x10 downsampled map (0-9)

        Returns:
            tuple[str, list[str]]: Status message and sequence of movements
        """
        # Get collision map, terrain, and sprites
        collision_map = self.pyboy.game_wrapper.game_area_collision()
        terrain = self._downsample_array(collision_map)
        sprite_locations = self.get_sprites()

        # Get full map for tile values and current tileset
        full_map = self.pyboy.game_wrapper._get_screen_background_tilemap()
        reader = PokemonRedReader(self.pyboy.memory)
        tileset = reader.read_tileset()

        # Start at player position (always 4,4 in the 9x10 grid)
        start = (4, 4)
        end = (target_row, target_col)

        # Validate target position
        if not (0 <= target_row < 9 and 0 <= target_col < 10):
            return "Invalid target coordinates", []

        # A* algorithm
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, end)}

        # Track closest reachable point
        closest_point = start
        min_distance = heuristic(start, end)

        def reconstruct_path(current):
            path = []
            while current in came_from:
                prev = came_from[current]
                if prev[0] < current[0]:
                    path.append("down")
                elif prev[0] > current[0]:
                    path.append("up")
                elif prev[1] < current[1]:
                    path.append("right")
                else:
                    path.append("left")
                current = prev
            path.reverse()
            return path

        while open_set:
            _, current = heapq.heappop(open_set)

            # Check if we've reached target
            if current == end:
                path = reconstruct_path(current)
                is_wall = terrain[end[0]][end[1]] == 0
                if is_wall:
                    return (
                        f"Partial Success: Your target location is a wall. In case this is intentional, attempting to navigate there.",
                        path,
                    )
                else:
                    return (
                        f"Success: Found path to target at ({target_row}, {target_col}).",
                        path,
                    )

            # Track closest point
            current_distance = heuristic(current, end)
            if current_distance < min_distance:
                closest_point = current
                min_distance = current_distance

            # If we're next to target and target is a wall, we can end here
            if (abs(current[0] - end[0]) + abs(current[1] - end[1])) == 1 and terrain[
                end[0]
            ][end[1]] == 0:
                path = reconstruct_path(current)
                # Add final move onto wall
                if end[0] > current[0]:
                    path.append("down")
                elif end[0] < current[0]:
                    path.append("up")
                elif end[1] > current[1]:
                    path.append("right")
                else:
                    path.append("left")
                return (
                    f"Success: Found path to position adjacent to wall at ({target_row}, {target_col}).",
                    path,
                )

            # Check all four directions
            for dr, dc, direction in [
                (1, 0, "down"),
                (-1, 0, "up"),
                (0, 1, "right"),
                (0, -1, "left"),
            ]:
                neighbor = (current[0] + dr, current[1] + dc)

                # Check bounds
                if not (0 <= neighbor[0] < 9 and 0 <= neighbor[1] < 10):
                    continue
                # Skip walls unless it's the final destination
                if terrain[neighbor[0]][neighbor[1]] == 0 and neighbor != end:
                    continue
                # Skip sprites unless it's the final destination
                if (neighbor[1], neighbor[0]) in sprite_locations and neighbor != end:
                    continue

                # Check tile pair collisions
                # Get bottom-left tile of each 2x2 block
                current_tile = full_map[current[0] * 2 + 1][
                    current[1] * 2
                ]  # Bottom-left tile of current block
                neighbor_tile = full_map[neighbor[0] * 2 + 1][
                    neighbor[1] * 2
                ]  # Bottom-left tile of neighbor block
                if not self._can_move_between_tiles(
                    current_tile, neighbor_tile, tileset
                ):
                    continue

                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # If target unreachable, return path to closest point
        if closest_point != start:
            path = reconstruct_path(closest_point)
            return (
                f"Partial Success: Could not reach the exact target, but found a path to the closest reachable point.",
                path,
            )

        return (
            "Failure: No path is visible to the chosen location. You may need to explore a totally different path to get where you're trying to go.",
            [],
        )

    def get_state_from_memory(self) -> str:
        """
        Reads the game state from memory and returns a string representation of it.
        """
        reader = PokemonRedReader(self.pyboy.memory)
        memory_str = "# Current Game State\n\nThis information is direct from the emulator at the present moment along with your screenshot. Use the information below to make decisions about what to do and where to go next.\n\n"

        name = reader.read_player_name()
        if name == "NINTEN":
            name = "Not yet set"
        rival_name = reader.read_rival_name()
        if rival_name == "SONY":
            rival_name = "Not yet set"

        # Get valid moves
        valid_moves = self.get_valid_moves()
        valid_moves_str = ", ".join(valid_moves) if valid_moves else "None"

        # Present each field as a clear bullet for easier parsing by the LLM
        memory_str += f"- Player: {name}\n"
        # memory_str += f"- Rival: {rival_name}\n"
        # memory_str += f"- Money: ${reader.read_money()}\n"
        memory_str += f"- Current Environment: {reader.read_location()}\n"
        memory_str += f"- Coordinates: {reader.read_coordinates()}\n"
        # (No longer exposing valid‑moves list directly; model must infer from screenshot.)
        # memory_str += f"Badges: {', '.join(reader.read_badges())}\n"

        # Inventory
        # memory_str += "Inventory:\n"
        # for item, qty in reader.read_items():
        #     memory_str += f"  {item} x{qty}\n"

        # Dialog
        dialog = reader.read_dialog()
        if dialog:
            memory_str += f"Dialog: {dialog}\n"
        else:

            memory_str += "Dialog: None\n"

        # --------------------------------------------------------------
        # Door / warp hints (experimental)
        # --------------------------------------------------------------
        door_info = self._get_doors_info()
        if door_info:
            memory_str += (
                "\n# Available Doors And Warps\n\n"
                "Here is the list of doors/warps visible in this environment and their coordinates. "
                "You can navigate to one by calling navigate_to with that (row, col) or by manually pressing D‑pad moves until you reach it.\n"
            )
            for dest, (x, y) in door_info:
                if dest:
                    memory_str += f"- Visible Door, Stairs, or Warp located at ({x}, {y})\n"
                else:
                    memory_str += f"- Door / warp at ({x}, {y})\n"

        # Party Pokemon
        # memory_str += "\nPokemon Party:\n"
        # for pokemon in reader.read_party_pokemon():
        #     memory_str += f"\n{pokemon.nickname} ({pokemon.species_name}):\n"
        #     memory_str += f"Level {pokemon.level} - HP: {pokemon.current_hp}/{pokemon.max_hp}\n"
        #     memory_str += f"Types: {pokemon.type1.name}{', ' + pokemon.type2.name if pokemon.type2 else ''}\n"
        #     for move, pp in zip(pokemon.moves, pokemon.move_pp, strict=True):
        #         memory_str += f"- {move} (PP: {pp})\n"
        #     if pokemon.status != StatusCondition.NONE:
        #         memory_str += f"Status: {pokemon.status.get_status_name()}\n"

        return memory_str

    def stop(self):
        self.pyboy.stop()