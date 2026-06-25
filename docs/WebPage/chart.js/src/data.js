export const fetchBenchData = async () => {
    const res = await fetch("bench_results.json");

    if(!res.ok)
        throw new Error(`Failed to load bench_results.json : ${res.status}`);

    return res.json();
};