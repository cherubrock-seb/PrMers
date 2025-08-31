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
echo "=== Gerbicz Li error injection tests ==="
for erriter in 55; do
  echo -n "Testing ./prmers 9941 -erroriter $erriter... "
  output=$(./prmers 9941 -erroriter "$erriter" --noask -prp 2>&1)
  echo "$output" > "logs/gerbicz_error_9941_iter${erriter}.log"
  if echo "$output" | grep -q "Injected error at iteration 55" \
     && echo "$output" | grep -q "\[Gerbicz Li\] Check FAILED! iter=9940" \
     && echo "$output" | grep -q "\[Gerbicz Li\] Restore iter=0 (j=9940)"; then
    echo "✅"
  else
    echo "❌ Output mismatch (see logs/gerbicz_error_9941_iter${erriter}.log)"
    exit 1
  fi
done
for erriter in 9940; do
  echo -n "Testing ./prmers 9941 -erroriter $erriter... "
  output=$(./prmers 9941 -erroriter "$erriter" --noask -prp 2>&1)
  echo "$output" > "logs/gerbicz_error_9941_iter${erriter}.log"
  if echo "$output" | grep -q "Injected error at iteration $erriter" \
     && echo "$output" | grep -q "\[Gerbicz Li\] Check FAILED! iter=9940" \
     && echo "$output" | grep -q "\[Gerbicz Li\] Restore iter=0 (j=9940)"; then
    echo "✅"
  else
    echo "❌ Output mismatch (see logs/gerbicz_error_9941_iter${erriter}.log)"
    exit 1
  fi
done
echo ""
echo "=== Extended P-1 factoring tests ==="

declare -a pm1_tests=(
  "139 -pm1 -b1 457:P-1 factor stage 1 found: 5625767248687"
  "139 -pm1 -b1 192:No P-1 (stage 1) factor up to B1=192"
  "139 -pm1 -b1 192 -b2 457:No factor P-1 (stage 2) until B2 = 457"
  "139 -pm1 -b1 193 -b2 457:>>>  Factor P-1 (stage 2) found : 5625767248687"
  "239 -pm1 -b1 193 -b2 2503:P-1 factor stage 1 found: 927239786567617|>>>  Factor P-1 (stage 2) found : 124250696089090697678753"
  "263 -pm1 -b1 3527 -b2 16477:P-1 factor stage 1 found: 23671|>>>  Factor P-1 (stage 2) found : 321269073670148767"
  "367 -pm1 -b1 11981 -b2 38971:P-1 factor stage 1 found: 646300400639|>>>  Factor P-1 (stage 2) found : 50500996776315830904406967"
  "569 -pm1 -b1 9 -b2 677:>>>  Factor P-1 (stage 2) found : 55470673"
  "1097 -pm1 -b1 3 -b2 709:>>>  Factor P-1 (stage 2) found : 4576661533441"
  "2151 -pm1 -b1 256 -b2 4073:P-1 factor stage 1 found: 327405968242246366421788399|>>>  Factor P-1 (stage 2) found : 31810015665526476520196715312101168065463218256802641"
  "4133 -pm1 -b1 23 -b2 2099:>>>  Factor P-1 (stage 2) found : 11173615097"
  "44159 -pm1 -b1 23 -b2 31:No P-1 (stage 1) factor up to B1=23|>>>  Factor P-1 (stage 2) found : 1511297617"
  "144139 -pm1 -b1 3 -b2 3583:No P-1 (stage 1) factor up to B1=3|>>>  Factor P-1 (stage 2) found : 3098700223"
  "544139 -pm1 -b1 3 -b2 7:P-1 factor stage 1 found: 22853839|>>>  Factor P-1 (stage 2) found : 22853839"
  "11544157 -pm1 -b1 19 -b2 101:No P-1 (stage 1) factor up to B1=19|>>>  Factor P-1 (stage 2) found : 44306474567"
)


for test in "${pm1_tests[@]}"; do
  IFS=':' read -r args expected <<< "$test"
  echo -n "Testing ./prmers $args ... "
  output=$(./prmers $args --noask 2>&1)
  echo "$output" > "logs/pm1_${args// /_}.log"
  valid=true
  IFS='|' read -ra expected_lines <<< "$expected"
  for expected_line in "${expected_lines[@]}"; do
    if ! grep -qF "$expected_line" <<< "$output"; then
      echo "❌ Missing '$expected_line' (see logs/pm1_${args// /_}.log)"
      valid=false
      break
    fi
  done
  $valid && echo "✅"
done

echo ""
echo "=== Optimization flags tests ==="
for entry in "${optflags[@]}"; do
  opt=${entry%%:*}
  flag=${entry#*:}
  echo -n "Testing -O $opt... "
  output=$(./prmers 127 --noask -O "$opt" -prp -debug 2>&1)
  if echo "$output" | grep -F -- "$flag" >/dev/null; then
    echo "✅"
  else
    echo "❌ Missing '$flag'"
    exit 1
  fi
done

echo ""
echo "=== Out-of-range exponent verification ==="
p=5650242870
echo -n "Testing M${p} (should be rejected)… "
./prmers "$p" --noask -prp > "logs/bad_${p}.log" 2>&1
exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "❌ Unexpected success"
    exit 1
fi
if grep -q "Error: Exponent must be <= 5650242869" "logs/bad_${p}.log"; then
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
echo ""
echo "=== Proof generation verification for exponent 9941 ==="
output=$(./prmers 9941 --noask --proofile 2>&1)
echo "$output" > logs/proofs_9941.log
if echo "$output" | grep -q 'proof \[0\] : M 87f3d3eabe4d6049, h 4526397be82cea45' \
   && echo "$output" | grep -q 'proof \[1\] : M d6a355de518574d7, h 7faf92dd48dc2013' \
   && echo "$output" | grep -q 'proof \[2\] : M 5aac235405ca84c7, h 934611f5f1192dd0'; then
    if echo "$output" | grep -q 'Verification result: SUCCESS'; then
        echo "✅ Proofs for M9941 generated and verified"
    else
        echo "❌ Proofs for M9941: verification failed (see logs/proofs_9941.log)"
        exit 1
    fi
else
    echo "❌ Proofs for M9941: unexpected output (see logs/proofs_9941.log)"
    exit 1
fi
echo ""
echo "=== P-1 factoring test for M541 ==="
output=$(./prmers 541 -pm1 -b1 899 --noask 2>&1)
echo "$output" > logs/pm1_541.log
if echo "$output" | grep -q 'P-1 factor stage 1 found: 4312790327'; then
    echo "✅ M541 P-1 factor found"
else
    echo "❌ M541 P-1 factor not found or output mismatch (see logs/pm1_541.log)"
    exit 1
fi

echo ""
echo "=== Mersenne cofactor PRP tests ==="

echo -n "Testing M2699 cofactor with 4 factors (composite)... "
./prmers -noask -prp 2699 -factors 5399,307687,1187561,7570504839257 > "logs/cofactor_composite_2699.log" 2>&1  
if [ $? -eq 0 ]; then
    echo "❌ Unexpected success (should be composite)"
    exit 1
else
    echo "✅"
fi

echo -n "Testing M2699 cofactor with 5 factors (PRP)... "
./prmers -noask -prp 2699 -factors 5399,307687,1187561,7570504839257,1987104667810711 > "logs/cofactor_prime_2699.log" 2>&1
if [ $? -ne 0 ]; then
    echo "❌ Failed (see logs/cofactor_prime_2699.log)"
    exit 1
else
    echo "✅"
fi

echo -e "\n🎉 All tests passed."