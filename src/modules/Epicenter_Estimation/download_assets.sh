#/bin/bash
# sacs_acc="https://drive.usercontent.google.com/download?id=1qHQfx5VxMxnBnKPJE9uZS8az2J5mfEeZ&confirm=y"
# sacs_vel="https://drive.usercontent.google.com/download?id=1XsueptPtGblvs1mes5oHMY_QEcsu7cMK&confirm=y"
sacs_acc="https://drive.usercontent.google.com/download?id=1Gx1LTKSTPGHTNYJKnewNCK_qK2U5JQF0&confirm=y"
download_file() {
  echo "Downloading $2 from $1"
  curl $1 --output $2
  echo "OK"
}
download_file "$sacs_acc" "sacs_acc.zip"
# download_file "$sacs_vel" "sacs_vel.zip"
unzip ./sacs_acc.zip
# unzip ./sacs_vel.zip
mv ./merge ./data/sacs/acc