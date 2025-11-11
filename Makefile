.SILENT:
MAKEFLAGS += --no-print-directory --silent

PYTHON        ?= python3

# Virtual environment
VENV_DIR      := .venv
VENV_PY       := $(VENV_DIR)/bin/python
REQUIREMENTS  := requirements.txt

# OSRM bits
OSRM_PORT     ?= 5001
OSRM_PROFILE  ?= $(shell brew --prefix osrm-backend)/share/osrm/profiles/car.lua

RAW_OSM_NAME  := canada-251027.osm.pbf
RAW_OSM       := data/raw/$(RAW_OSM_NAME)

# Use data/osrm as the OSRM working directory (matches where your .osrm.* files already live)
OSRM_DIR      := data/osrm
OSRM_BASE     := $(OSRM_DIR)/canada-251027.osrm
OSRM_OSRM     := $(OSRM_DIR)/canada-251027.osrm
OSRM_PART     := $(OSRM_DIR)/canada-251027.osrm.partition
OSRM_CELLS    := $(OSRM_DIR)/canada-251027.osrm.cells

# Project data
REGIONS_GEOJSON := data/raw/Canada-20-subdivisions.geojson
ROADS_GJ        := data/network/roads_ferries_drivable.geojsonl
BORDERS_CSV     := data/crossings/cleaned_up_border_crossings.csv
STARTS_CSV      := data/config/starts.csv
AVG_TIMES_CSV   := output/avg_times_by_region.csv

# ----------------------------
# Phony targets
# ----------------------------

.PHONY: all dirs osrm-prep osrm-all osrm osrm-forever borders times clean venv

# Default: build borders + times (assumes OSRM server is already running)
all: borders times

# Ensure directory structure exists
dirs:
	mkdir -p scripts data/raw data/network data/config data/crossings output data/osrm

# ----------------------------
# Virtual env + deps
# ----------------------------

venv: $(VENV_PY)

# ensure venv exists and requirements installed
$(VENV_PY): $(REQUIREMENTS)
	@$(PYTHON) -m venv $(VENV_DIR) >/dev/null 2>&1
	@$(VENV_PY) -m pip install --upgrade pip >/dev/null 2>&1
	@if [ -s $(REQUIREMENTS) ]; then \
		$(VENV_PY) -m pip install -r $(REQUIREMENTS) >/dev/null 2>&1; \
	fi

# ----------------------------
# OSRM: prepare graph data
# ----------------------------

# Full pipeline: extract -> partition -> customize
osrm-prep: $(OSRM_CELLS)
	@echo "[osrm] Graph prepared at $(OSRM_BASE).*"

# Convenience target: prepares all OSRM data and confirms ready
osrm-all: osrm-prep
	@echo "[osrm] OSRM dataset ready at $(OSRM_OSRM)"

# Extract base .osrm from PBF (and sidecar .osrm.* files) into data/raw, then move into data/osrm
$(OSRM_OSRM): $(RAW_OSM) | dirs
	@echo "[osrm] Extracting from $(RAW_OSM) using profile $(OSRM_PROFILE)..."
	# 1) Let OSRM write all .osrm* next to the PBF in data/raw
	osrm-extract -p "$(OSRM_PROFILE)" "$(RAW_OSM)"
	# 2) Move all generated canada-251027.osrm* files into data/osrm
	@mkdir -p "$(OSRM_DIR)"
	mv data/raw/canada-251027.osrm* "$(OSRM_DIR)/"

# Partition for MLD
$(OSRM_PART): $(OSRM_OSRM)
	@echo "[osrm] Partitioning $(OSRM_OSRM)..."
	osrm-partition "$(OSRM_OSRM)"

# Customize for MLD
$(OSRM_CELLS): $(OSRM_PART)
	@echo "[osrm] Customizing $(OSRM_OSRM)..."
	osrm-customize "$(OSRM_OSRM)"
	@touch "$(OSRM_CELLS)"  # ensure Make sees the target exist

# ----------------------------
# OSRM: run server
# ----------------------------

# Run OSRM routed server (blocking). Use in a separate terminal:
#   make osrm
osrm: osrm-prep
	@echo "[osrm] Starting OSRM server on port $(OSRM_PORT)..."
	@echo "[osrm] Press Ctrl+C to stop."
	osrm-routed --algorithm mld -p $(OSRM_PORT) "$(OSRM_OSRM)"

# ----------------------------
# Boundary crossings
# ----------------------------

# Generate boundary_crossings.csv from drivable network
borders: $(VENV_PY) $(BORDERS_CSV)

$(BORDERS_CSV): $(ROADS_GJ) $(REGIONS_GEOJSON) scripts/find_boundry_crossings.py | dirs
	$(VENV_PY) scripts/find_boundry_crossings.py \
	  --regions "$(REGIONS_GEOJSON)" \
	  --section-field CNAME \
	  --roads "$(ROADS_GJ)" \
	  --out "$(BORDERS_CSV)"

# ----------------------------
# Travel times (OSRM-based)
# ----------------------------

# Always recompute travel times when you run `make times`
times: venv borders | dirs
	$(VENV_PY) scripts/travel_times_from_point.py \
	  --borders-csv "$(BORDERS_CSV)" \
	  --starts-csv "$(STARTS_CSV)" \
	  --osrm-url "http://127.0.0.1:$(OSRM_PORT)" \
	  --out "$(AVG_TIMES_CSV)" \
	  --section-field CDNAME \
	  --regions data/raw/Canada-20-subdivisions.geojson

# ----------------------------
# Cleanup
# ----------------------------

clean:
	@echo "[clean] Removing generated outputs (not raw data or OSRM graph)..."
	rm -f "$(BORDERS_CSV)" "$(AVG_TIMES_CSV)" output/times_from_*_border.csv
