#!/bin/bash

mkdir -p logs

prime_exponents=(
    89 107 127 521 607 1279 2203 2281
    3217 4253 4423 9689 9941 11213 19937
    21701 23209 44497 86243 110503 132049
    216091
)

composite_exponents=(
    57 91 100 200 500 1001 4095 8191
)
optflags=(
  fastmath:-cl-fast-relaxed-math
  mad:-cl-mad-enable
  unsafe:-cl-unsafe-math-optimizations
  nans:-cl-no-signed-zeros
  optdisable:-cl-opt-disable
)

echo ""
echo "=== Optimization flags tests ==="
for entry in "${optflags[@]}"; do
  opt=${entry%%:*}
  flag=${entry#*:}
  echo -n "Testing -O $opt... "
  output=$(./prmers 127 --noask -O "$opt" -prp 2>&1)
  if echo "$output" | grep -F -- "$flag" >/dev/null; then
    echo "✅"
  else
    echo "❌ Missing '$flag'"
    exit 1
  fi
done

echo ""
echo "=== Combined flags via '-O fastmath mad unsafe nans optdisable' ==="
combined_flags=( -cl-fast-relaxed-math -cl-mad-enable -cl-unsafe-math-optimizations -cl-no-signed-zeros -cl-opt-disable )
output=$(./prmers 127 --noask -O fastmath mad unsafe nans optdisable -prp 2>&1)
for flag in "${combined_flags[@]}"; do
  echo -n "  checking $flag... "
  if echo "$output" | grep -F -- "$flag" >/dev/null; then
    echo "✅"
  else
    echo "❌ Missing '$flag'"
    exit 1
  fi
done

echo ""
echo "=== Out-of-range exponent verification ==="
p=1300000000
echo -n "Testing M${p} (should be rejected)… "
./prmers "$p" --noask -prp > "logs/bad_${p}.log" 2>&1
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "❌ Unexpected success"
    exit 1
fi
if grep -q "Error: Exponent must be <= 1207959503" "logs/bad_${p}.log"; then
    echo "✅ Correctly rejected (see logs/bad_${p}.log)"
else
    echo "❌ Missing or wrong error message (see logs/bad_${p}.log)"
    exit 1
fi

echo ""
echo "=== Prime exponents ==="
for p in "${prime_exponents[@]}"; do
    echo -n "Testing M$p... "
    ./prmers "$p" --noask -prp > "logs/ok_${p}.log" 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Failed (see logs/ok_${p}.log)"
        exit 1
    else
        echo "✅"
    fi
done

echo ""
echo "=== Composite exponents ==="
for p in "${composite_exponents[@]}"; do
    echo -n "Testing M$p (composite)... "
    ./prmers "$p" --noask -ll > "logs/fail_${p}.log" 2>&1
    if [ $? -eq 0 ]; then
        echo "❌ Unexpected success"
        exit 1
    else
        echo "✅"
    fi
done

echo ""
echo "=== Specific result verification ==="

# Test for exponent 100003
echo -n "Testing M100003… and check res64 and res2048"
output=$(./prmers 100003 --noask -prp 2>&1)
echo "$output" > logs/ok_100003.log
if echo "$output" | grep -q '"res64":"1CF45E9503C71FD6"' \
   && echo "$output" | grep -q '"res2048":"af262d00ed00a05d53e99d0e0e451b12405ddabe139fe8396a4c520b505bb65bed1609d3c8ef23bbb1d0f8140a6bcdd2c67f9c8aa3bd0e6eeb3e8e79db904810c88de09820557176b389290f84f18424efa6a59fb9f132a74f53a83ba6e2f508c617a5e1451c3ee08d179e6614026f973d1900602f2068a08894cd81ed5035de9ded85909b1ee6ff4dc723118b79d3f940272ae1066aebe27c86338ad7edf70e76c0e8abf3e985b73db2a06f1b742a9a908728be2bd4b7daa2d6aafc11bacaaa40944e9a66b039cb0deaaa8e5e357cd54b81b3ec6661d55e48bacb994bfd3cbb33f3f01d82347fa00578ec86c4cd7eb568a1463cf3e38dae1cf45e9503c71fd6"' 
then
    echo "✅ M100003 OK"
else
    echo "❌ M100003: unexpected output (see logs/ok_100003.log)"
    exit 1
fi

# Test for exponent 11213
echo -n "Testing M11213… and check res64 and res2048"
output=$(./prmers 11213 --noask -prp 2>&1)
echo "$output" > logs/ok_11213.log
if echo "$output" | grep -q '"res64":"0000000000000001"' \
   && echo "$output" | grep -q '"res2048":"00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001"' 
then
    echo "✅ M11213 OK"
else
    echo "❌ M11213: unexpected output (see logs/ok_11213.log)"
    exit 1
fi

echo ""
echo "=== Intermediate res64 display verification ==="
output=$(./prmers 11213 --noask -prp -res64_display_interval 1000 2>&1)
echo "$output" > logs/res64_interval_11213.log
expected_lines=(
  "Iter: 1000| Res64: FBA631FBCB73A011"
  "Iter: 2000| Res64: F01283650C4A1491"
  "Iter: 3000| Res64: 7E79193B757010B7"
  "Iter: 4000| Res64: 31482E4D80FE99BB"
  "Iter: 5000| Res64: 973B76BACF73BBEF"
  "Iter: 6000| Res64: 8CFFB332495FC320"
  "Iter: 7000| Res64: 98080C76DF068843"
  "Iter: 8000| Res64: 8FDA516F885D3FEE"
  "Iter: 9000| Res64: 2AADBC4F1E318E92"
  "Iter: 10000| Res64: 0A4AAF339C8B290C"
  "Iter: 11000| Res64: A1F26F470CFE412D"
)

for line in "${expected_lines[@]}"; do
  if ! grep -qF "$line" <<< "$output"; then
    echo "❌ Missing intermediate line: '$line' (see logs/res64_interval_11213.log)"
    exit 1
  fi
done

echo "✅ Intermediate res64 values OK"


echo -e "\n🎉 All tests passed."
