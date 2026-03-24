#!/bin/bash

set -o pipefail

# ==========================================
# XAC DINH VI TRI CUA SCRIPT
# ==========================================
# Lay duong dan thu muc chua chinh file run.sh nay
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# ==========================================
# CAU HINH THONG TIN (Ban chi can sua phan nay)
# ==========================================
LOCAL_DIR="/media/samer/DATA1/DT/Code/PJ/BadgeGuard/"                 # Thu muc o may ban (NHO CO DAU / O CUOI)
REMOTE_USER="aidev"     # Vi du: root hoac ubuntu
REMOTE_HOST="aiserver.daotao.ai"    # Vi du: 192.168.1.10
REMOTE_PORT="22"               # Port SSH (Mac dinh la 22)
REMOTE_DIR="/home/aidev/workspace/bg/BadgeGuard/"  # Thu muc tren Server (VD: /var/www/html/my-app/)


REMOTE_PULL_DIR="/home/aidev/workspace/bg/BadgeGuard/"  # Thu muc muon tai tu server
LOCAL_PULL_DIR="/media/samer/DATA1/DT/Code/PJ/BadgeGuard/"  # Thu muc dich tren may ca nhan

# Duong dan file ignore rieng cho tung che do
EXCLUDE_FILE_PUSH="$SCRIPT_DIR/.rsyncignore.push"
INCLUDE_FILE_PUSH="$SCRIPT_DIR/.rsyncinclude.push"
EXCLUDE_FILE_PULL="$SCRIPT_DIR/.rsyncignore.pull"
INCLUDE_FILE_PULL="$SCRIPT_DIR/.rsyncinclude.pull"
SSH_CONTROL_PATH="/tmp/rsync-ctrl-${REMOTE_USER}-${REMOTE_HOST}-${REMOTE_PORT}"
GENERATED_INCLUDE_FILE=""

# ==========================================
# HE THONG XU LY (Khong can sua phan duoi nay)
# ==========================================
print_header() {
    echo "════════════════════════════════════════"
    echo " $1"
}

print_info() {
    echo -e "  ${C_INFO}[INFO]${C_RESET} $1"
}

print_ok() {
    echo -e "  ${C_OK}[OK]${C_RESET}   $1"
}

print_warn() {
    echo -e "  ${C_WARN}[WARN]${C_RESET} $1"
}

print_error() {
    echo -e "  ${C_ERR}[ERR]${C_RESET}  $1"
}

print_kv() {
    local key="$1"
    local value="$2"
    printf "  %-8s %s\n" "$key:" "$value"
}

print_step() {
    local number="$1"
    local text="$2"
    echo -e "\n${C_ACCENT}[BUOC $number]${C_RESET} $text"
}

C_RESET=""
C_INFO=""
C_OK=""
C_WARN=""
C_ERR=""
C_ACCENT=""

if [[ -t 1 ]]; then
    C_RESET='\033[0m'
    C_INFO='\033[36m'
    C_OK='\033[32m'
    C_WARN='\033[33m'
    C_ERR='\033[31m'
    C_ACCENT='\033[35m'
fi

if ! command -v rsync >/dev/null 2>&1; then
    print_error "Khong tim thay lenh rsync. Vui long cai rsync truoc khi chay script."
    print_info "Ubuntu/Debian: sudo apt install rsync"
    exit 1
fi

build_include_filter() {
    local source_file="$1"
    local line=""
    local normalized=""
    local part=""
    local current=""
    local root=""
    local is_dir="0"
    local seen="0"
    local -a parts=()
    local -a roots=()

    GENERATED_INCLUDE_FILE="$(mktemp /tmp/rsync-include.XXXXXX)" || return 1
    : > "$GENERATED_INCLUDE_FILE"

    while IFS= read -r line || [ -n "$line" ]; do
        normalized="$line"
        normalized="${normalized#"${normalized%%[![:space:]]*}"}"
        normalized="${normalized%"${normalized##*[![:space:]]}"}"

        if [ -z "$normalized" ] || [[ "$normalized" == \#* ]]; then
            continue
        fi

        if [[ "$normalized" == [+-][[:space:]]* ]]; then
            normalized="${normalized:1}"
            normalized="${normalized#"${normalized%%[![:space:]]*}"}"
        fi

        normalized="${normalized#./}"
        while [[ "$normalized" == *"//"* ]]; do
            normalized="${normalized//\/\//\/}"
        done

        if [[ "$normalized" == /* ]]; then
            print_warn "Bo qua path tuyet doi trong file ngoai le: $normalized"
            continue
        fi

        is_dir="0"
        if [[ "$normalized" == */ ]]; then
            is_dir="1"
            normalized="${normalized%/}"
        fi

        if [ -z "$normalized" ]; then
            continue
        fi

        IFS='/' read -r -a parts <<< "$normalized"
        current=""
        for part in "${parts[@]}"; do
            if [ -z "$part" ]; then
                continue
            fi

            if [ -z "$current" ]; then
                current="$part"
            else
                current="$current/$part"
            fi

            printf "+ %s/\n" "$current" >> "$GENERATED_INCLUDE_FILE"
        done

        if [ "$is_dir" = "1" ]; then
            printf "+ %s/**\n" "$normalized" >> "$GENERATED_INCLUDE_FILE"
        else
            printf "+ %s\n" "$normalized" >> "$GENERATED_INCLUDE_FILE"
        fi

        root="${parts[0]}"
        if [ -n "$root" ]; then
            seen="0"
            for part in "${roots[@]}"; do
                if [ "$part" = "$root" ]; then
                    seen="1"
                    break
                fi
            done

            if [ "$seen" = "0" ]; then
                roots+=("$root")
            fi
        fi
    done < "$source_file"

    if [ "${#roots[@]}" -eq 0 ]; then
        rm -f "$GENERATED_INCLUDE_FILE" >/dev/null 2>&1 || true
        GENERATED_INCLUDE_FILE=""
        return 1
    fi

    for root in "${roots[@]}"; do
        printf -- "- %s/**\n" "$root" >> "$GENERATED_INCLUDE_FILE"
    done

    return 0
}

cleanup_generated_include_file() {
    if [ -n "$GENERATED_INCLUDE_FILE" ]; then
        rm -f "$GENERATED_INCLUDE_FILE" >/dev/null 2>&1 || true
    fi
}

cleanup_ssh_master() {
    ssh -p "$REMOTE_PORT" -o ControlPath="$SSH_CONTROL_PATH" -O exit "${REMOTE_USER}@${REMOTE_HOST}" >/dev/null 2>&1 || true
    rm -f "$SSH_CONTROL_PATH" >/dev/null 2>&1 || true
}

setup_ssh_master() {
    rm -f "$SSH_CONTROL_PATH" >/dev/null 2>&1 || true
    if ssh -p "$REMOTE_PORT" \
        -o ControlMaster=yes \
        -o ControlPersist=10m \
        -o ControlPath="$SSH_CONTROL_PATH" \
        -o BatchMode=no \
        -Nf "${REMOTE_USER}@${REMOTE_HOST}"; then
        true
    else
        print_error "Khong the xac thuc SSH. Dung script."
        print_info "Goi y: kiem tra mat khau, host, user, port hoac ket noi mang."
        exit 1
    fi
}

trap 'cleanup_generated_include_file; cleanup_ssh_master' EXIT

setup_ssh_master

print_header "CHON CHE DO DONG BO"
echo "  1) Day du lieu tu may len server"
echo "  2) Tai du lieu tu server ve may"
echo "  3) Dung lai"

SYNC_MODE=""
while true; do
    read -rp "Nhap lua chon cua ban (1/2/3): " mode_choice
    case $mode_choice in
        1)
            SYNC_MODE="push"
            break
            ;;
        2)
            SYNC_MODE="pull"
            break
            ;;
        3)
            # print_warn "Da huy bo thao tac."
            exit 0
            ;;
        *)
            print_warn "Lua chon khong hop le. Vui long nhap 1, 2 hoac 3."
            ;;
    esac
done

if [ "$SYNC_MODE" = "push" ]; then
    if [ ! -d "$LOCAL_DIR" ]; then
        print_error "Thu muc nguon khong ton tai: $LOCAL_DIR"
        exit 1
    fi

    ACTION_TITLE="CHUAN BI DAY DU LIEU LEN SERVER"
    ACTION_START="BAT DAU DAY DU LIEU"
    ACTION_DONE="HOAN TAT DAY DU LIEU!"
    ACTION_FAIL="Day du lieu that bai. Vui long kiem tra ket noi SSH/duong dan."
    SRC_PATH="$LOCAL_DIR"
    DEST_PATH="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
else
    if ! mkdir -p "$LOCAL_PULL_DIR"; then
        print_error "Khong the tao thu muc dich tren may: $LOCAL_PULL_DIR"
        exit 1
    fi

    ACTION_TITLE="CHUAN BI TAI DU LIEU TU SERVER VE MAY"
    ACTION_START="BAT DAU TAI DU LIEU"
    ACTION_DONE="HOAN TAT TAI DU LIEU!"
    ACTION_FAIL="Tai du lieu that bai. Vui long kiem tra ket noi SSH/duong dan."
    SRC_PATH="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PULL_DIR}"
    DEST_PATH="$LOCAL_PULL_DIR"
fi

print_header "$ACTION_TITLE"
print_kv "Nguon" "$SRC_PATH"
print_kv "Dich" "$DEST_PATH"
print_kv "SSH" "port $REMOTE_PORT"

# Khoi tao mang chua cac tham so rsync
SSH_CMD="ssh -p $REMOTE_PORT -o ControlMaster=auto -o ControlPersist=10m -o ControlPath=$SSH_CONTROL_PATH"
OPTIONS=(-avzP -e "$SSH_CMD")
# print_info "SSH se tai su dung ket noi trong 10 phut de han che nhap lai mat khau."

if [ "$SYNC_MODE" = "push" ]; then
    print_header "CO DUNG FLAG --delete KHONG?"
    echo "  1) Co (Xoa file tren Server neu o May khong co)"
    echo "  2) Khong (Giu lai file tren Server)"
    while true; do
        read -rp "Nhap lua chon cua ban (1/2): " del_choice
        case $del_choice in
            1)
                OPTIONS+=(--delete)
                print_ok "Da bat co --delete"
                break
                ;;
            2)
                print_ok "Da bo qua --delete"
                break
                ;;
            *)
                print_warn "Lua chon khong hop le. Vui long nhap 1 hoac 2."
                ;;
        esac
    done
    if [ -f "$INCLUDE_FILE_PUSH" ]; then
        if build_include_filter "$INCLUDE_FILE_PUSH"; then
            OPTIONS+=(--filter="merge $GENERATED_INCLUDE_FILE")
            print_ok "  Da nhan dien file ngoai le cho push tai: $INCLUDE_FILE_PUSH"
        else
            print_warn "File ngoai le push khong co rule hop le. Se bo qua ngoai le."
        fi
    else
        print_warn "Khong tim thay file ngoai le push tai $INCLUDE_FILE_PUSH. Se bo qua ngoai le."
    fi
    if [ -f "$EXCLUDE_FILE_PUSH" ]; then
        OPTIONS+=(--exclude-from="$EXCLUDE_FILE_PUSH")
        print_ok "  Da nhan dien file loai tru cho push tai: $EXCLUDE_FILE_PUSH"
    else
        print_warn "Khong tim thay file loai tru push tai $EXCLUDE_FILE_PUSH. Se dong bo tat ca."
    fi
else
    if [ -f "$INCLUDE_FILE_PULL" ]; then
        if build_include_filter "$INCLUDE_FILE_PULL"; then
            OPTIONS+=(--filter="merge $GENERATED_INCLUDE_FILE")
            print_ok "  Da nhan dien file ngoai le cho pull tai: $INCLUDE_FILE_PULL"
        else
            print_warn "File ngoai le pull khong co rule hop le. Se bo qua ngoai le."
        fi
    else
        print_warn "Khong tim thay file ngoai le pull tai $INCLUDE_FILE_PULL. Se bo qua ngoai le."
    fi
    if [ -f "$EXCLUDE_FILE_PULL" ]; then
        OPTIONS+=(--exclude-from="$EXCLUDE_FILE_PULL")
        print_ok "  Da nhan dien file loai tru cho pull tai: $EXCLUDE_FILE_PULL"
    else
        print_warn "Khong tim thay file loai tru pull tai $EXCLUDE_FILE_PULL. Se tai ve tat ca."
    fi
    print_info " Che do tai ve dung dung cu phap rsync: remote -> local (khong --delete)."
fi

print_header "DANG CHAY THU (DRY-RUN)"

if ! rsync "${OPTIONS[@]}" --dry-run "$SRC_PATH" "$DEST_PATH"; then
    print_error "Dry-run that bai. Dung script."
    print_info "Goi y: kiem tra SSH key, quyen truy cap, duong dan remote va port."
    exit 1
fi

echo "  1) Tiep tuc (chay THAT)"
echo "  2) Dung lai"

while true; do
    read -rp "Nhap lua chon cua ban (1/2): " choice
    case $choice in
        1)
            print_header "$ACTION_START"
            if rsync "${OPTIONS[@]}" "$SRC_PATH" "$DEST_PATH"; then
                print_ok "$ACTION_DONE"
                exit 0
            else
                print_error "$ACTION_FAIL"
                print_info "Ban co the chay lai script de xem dry-run va loi chi tiet."
                exit 1
            fi
            ;;
        2)
            # print_warn "Da huy bo thao tac."
            exit 0
            ;;
        *)
            print_warn "Lua chon khong hop le. Vui long nhap 1 hoac 2."
            ;;
    esac
done
