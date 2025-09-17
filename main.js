import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7/+esm'
import { COUNTRIES } from './countries.js'

const MAX_GUESSES = 10
const DAY_IN_MS = 24 * 60 * 60 * 1000
const REFERENCE_DATE = new Date(Date.UTC(2024, 0, 1))

const today = new Date()
const todayUTC = new Date(Date.UTC(today.getUTCFullYear(), today.getUTCMonth(), today.getUTCDate()))
const puzzleDateKey = todayUTC.toISOString().slice(0, 10)
let dayIndex = Math.floor((todayUTC - REFERENCE_DATE) / DAY_IN_MS)
if (!Number.isFinite(dayIndex)) dayIndex = 0
const safeIndex = ((dayIndex % COUNTRIES.length) + COUNTRIES.length) % COUNTRIES.length
const puzzleNumber = dayIndex + 1
const solution = COUNTRIES[safeIndex]

const stateKey = `gdpe-state-${puzzleDateKey}`
const storedState = (() => {
  try {
    const raw = localStorage.getItem(stateKey)
    if (!raw) return null
    return JSON.parse(raw)
  } catch (error) {
    console.warn('Failed to parse saved state', error)
    return null
  }
})()

const state = {
  status: storedState?.status ?? 'IN_PROGRESS',
  guesses: (storedState?.guesses ?? []).map((guess) => ({
    ...guess,
    distance: typeof guess.distance === 'number' ? guess.distance : Math.round(Number(guess.distance) || 0),
    directionSymbol: guess.directionSymbol ?? directionSymbol(guess.direction ?? '•')
  }))
}

const metaEl = document.getElementById('puzzle-meta')
const gdpTotalEl = document.getElementById('gdp-total')
const guessForm = document.getElementById('guess-form')
const guessInput = document.getElementById('guess-input')
const guessHistoryEl = document.getElementById('guess-history')
const feedbackEl = document.getElementById('feedback')
const guessCounterEl = document.getElementById('guess-counter')
const statusBannerEl = document.getElementById('status-banner')

const datalist = document.getElementById('country-list')
datalist.innerHTML = COUNTRIES.slice()
  .sort((a, b) => a.name.localeCompare(b.name))
  .map((country) => `<option value="${country.name}"></option>`)
  .join('')

metaEl.innerHTML = `
  <span>Puzzle #${puzzleNumber}</span>
  <span>${todayUTC.toLocaleDateString(undefined, { month: 'long', day: 'numeric', year: 'numeric' })}</span>
`

gdpTotalEl.textContent = `Total GDP: ${formatBillions(solution.gdp)} USD`

document.title = `GDPe — Daily GDP Puzzle #${puzzleNumber}`

renderTreemap(solution)
renderGuessState()
updateGuessCounter()
updateStatusBanner()
const lastGuess = state.guesses[state.guesses.length - 1]
if (state.status === 'WIN') {
  feedbackEl.textContent = successMessage(state.guesses.length)
} else if (state.status === 'LOSE') {
  feedbackEl.textContent = `The country was ${solution.name}.`
} else if (lastGuess) {
  feedbackEl.textContent = hintMessage(lastGuess)
}
updateInputState()

const resizeObserver = new ResizeObserver(() => {
  renderTreemap(solution)
})
resizeObserver.observe(document.querySelector('.treemap-wrapper'))

guessForm.addEventListener('submit', (event) => {
  event.preventDefault()
  if (state.status !== 'IN_PROGRESS') return

  const guessText = guessInput.value.trim()
  if (!guessText) {
    feedbackEl.textContent = 'Enter a country name to make a guess.'
    return
  }

  const matchedCountry = findCountry(guessText)
  if (!matchedCountry) {
    feedbackEl.textContent = 'That country is not in our list. Try another guess.'
    return
  }

  if (state.guesses.some((guess) => guess.iso2 === matchedCountry.iso2)) {
    feedbackEl.textContent = 'You already tried that country.'
    guessInput.value = ''
    return
  }

  const guessRecord = buildGuessRecord(matchedCountry)
  state.guesses.push(guessRecord)

  if (matchedCountry.iso2 === solution.iso2) {
    state.status = 'WIN'
    feedbackEl.textContent = successMessage(state.guesses.length)
  } else if (state.guesses.length >= MAX_GUESSES) {
    state.status = 'LOSE'
    feedbackEl.textContent = `No luck this time. The country was ${solution.name}.`
  } else {
    feedbackEl.textContent = hintMessage(guessRecord)
  }

  saveState()
  renderGuessState()
  updateGuessCounter()
  updateStatusBanner()
  updateInputState()
  guessInput.value = ''
})

function buildGuessRecord(country) {
  const distance = haversineDistance(country.lat, country.lon, solution.lat, solution.lon)
  const bearing = calculateBearing(country.lat, country.lon, solution.lat, solution.lon)
  const direction = bearingToCompass(bearing)
  return {
    name: country.name,
    iso2: country.iso2,
    distance: Math.round(distance),
    bearing,
    direction,
    directionSymbol: directionSymbol(direction),
    correct: country.iso2 === solution.iso2
  }
}

function renderGuessState() {
  guessHistoryEl.innerHTML = ''
  state.guesses.forEach((guess, index) => {
    const item = document.createElement('li')
    item.className = `guess-item${guess.correct ? ' correct' : ''}`
    const attempt = index + 1
    item.innerHTML = `
      <div class="guess-item__header">
        <span>Guess ${attempt}</span>
        <span>${guess.name}</span>
      </div>
      <div class="guess-item__details">
        <span class="direction-chip">${guess.directionSymbol} ${guess.direction}</span>
        <span class="distance-chip">${formatDistance(guess.distance)}</span>
      </div>
    `
    guessHistoryEl.append(item)
  })
}

function updateGuessCounter() {
  const remaining = MAX_GUESSES - state.guesses.length
  const guessWord = remaining === 1 ? 'guess' : 'guesses'
  guessCounterEl.textContent = `${remaining} ${guessWord} remaining`
}

function updateStatusBanner() {
  statusBannerEl.textContent = ''
  statusBannerEl.className = 'card__footer'
  if (state.status === 'WIN') {
    statusBannerEl.textContent = `Solved in ${state.guesses.length} ${state.guesses.length === 1 ? 'guess' : 'guesses'}!`
    statusBannerEl.classList.add('status-success')
  } else if (state.status === 'LOSE') {
    statusBannerEl.textContent = `Puzzle over — it was ${solution.name}.`
    statusBannerEl.classList.add('status-failure')
  }
}

function updateInputState() {
  const completed = state.status !== 'IN_PROGRESS'
  guessInput.disabled = completed
  guessForm.querySelector('button').disabled = completed
  if (completed) {
    guessInput.placeholder = state.status === 'WIN' ? 'Puzzle solved!' : `It was ${solution.name}.`
  }
}

function saveState() {
  try {
    localStorage.setItem(stateKey, JSON.stringify(state))
  } catch (error) {
    console.warn('Unable to save state', error)
  }
}

function findCountry(query) {
  const normalized = normalize(query)
  return COUNTRIES.find((country) => {
    if (normalize(country.name) === normalized) return true
    if (country.aliases?.some((alias) => normalize(alias) === normalized)) return true
    return false
  })
}

function normalize(value) {
  return value.trim().toLowerCase()
}

function hintMessage(guess) {
  return `${guess.name} is ${formatDistance(guess.distance)} away, head ${guess.direction}.`
}

function successMessage(attempts) {
  const attemptWord = attempts === 1 ? 'guess' : 'guesses'
  return `Correct! You nailed it in ${attempts} ${attemptWord}.`
}

function formatBillions(value) {
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 0
  }).format(value) + 'B'
}

function formatDistance(distance) {
  return `${distance.toLocaleString()} km`
}

function directionSymbol(direction) {
  const symbols = {
    N: '↑',
    NE: '↗︎',
    E: '→',
    SE: '↘︎',
    S: '↓',
    SW: '↙︎',
    W: '←',
    NW: '↖︎'
  }
  return symbols[direction] ?? '•'
}

function renderTreemap(country) {
  const wrapper = document.querySelector('.treemap-wrapper')
  const svg = d3.select('#treemap')
  const width = wrapper.clientWidth
  const height = wrapper.clientHeight
  svg.attr('width', width)
  svg.attr('height', height)
  svg.selectAll('*').remove()

  const hierarchy = d3
    .hierarchy({ name: 'GDP', children: country.breakdown })
    .sum((d) => d.value || 0)
    .sort((a, b) => b.value - a.value)

  const treemap = d3.treemap().size([width, height]).paddingInner(3).paddingOuter(2)
  treemap(hierarchy)

  const leaves = hierarchy.leaves()
  if (!leaves.length) return

  const values = leaves.map((leaf) => leaf.value)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const domainMin = min === max ? min - 1 : min
  const domainMax = min === max ? max + 1 : max
  const colorScale = d3
    .scaleSequential()
    .domain([domainMin, domainMax])
    .interpolator(d3.interpolateLab('hsl(96, 85%, 92%)', 'hsl(96, 85%, 45%)'))

  const cell = svg
    .selectAll('g')
    .data(leaves)
    .enter()
    .append('g')
    .attr('transform', (d) => `translate(${d.x0},${d.y0})`)

  cell
    .append('rect')
    .attr('class', 'treemap-rect')
    .attr('width', (d) => Math.max(0, d.x1 - d.x0))
    .attr('height', (d) => Math.max(0, d.y1 - d.y0))
    .attr('fill', (d) => colorScale(d.value))

  const labels = cell
    .append('text')
    .attr('class', 'treemap-label')
    .attr('x', 6)
    .attr('y', 14)

  labels.each(function (d) {
    const text = d3.select(this)
    const width = d.x1 - d.x0
    const height = d.y1 - d.y0
    const name = d.data.name
    const value = d.value

    if (width < 60 || height < 35) {
      text.text('')
      return
    }

    const words = name.split(/\s+/)
    let line = []
    let lineNumber = 0
    const lineHeight = 1.2
    const x = 6
    const y = 14
    let tspan = text.append('tspan').attr('x', x).attr('y', y)

    for (let i = 0; i < words.length; i++) {
      line.push(words[i])
      tspan.text(line.join(' '))
      if (tspan.node().getComputedTextLength() > width - 12) {
        line.pop()
        tspan.text(line.join(' '))
        line = [words[i]]
        lineNumber += 1
        if ((lineNumber + 1) * 12 > height) {
          text.selectAll('tspan').remove()
          return
        }
        tspan = text.append('tspan').attr('x', x).attr('y', y + lineNumber * 12).text(words[i])
      }
    }

    lineNumber += 1
    if ((lineNumber + 1) * 12 <= height) {
      text
        .append('tspan')
        .attr('x', x)
        .attr('y', y + lineNumber * 12)
        .attr('font-weight', 600)
        .text(formatBillions(Math.round(value)))
    }
  })
}

function haversineDistance(lat1, lon1, lat2, lon2) {
  const R = 6371
  const dLat = toRadians(lat2 - lat1)
  const dLon = toRadians(lon2 - lon1)
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2)) *
      Math.sin(dLon / 2) * Math.sin(dLon / 2)
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  return R * c
}

function calculateBearing(lat1, lon1, lat2, lon2) {
  const phi1 = toRadians(lat1)
  const phi2 = toRadians(lat2)
  const deltaLon = toRadians(lon2 - lon1)

  const y = Math.sin(deltaLon) * Math.cos(phi2)
  const x =
    Math.cos(phi1) * Math.sin(phi2) -
    Math.sin(phi1) * Math.cos(phi2) * Math.cos(deltaLon)
  const theta = Math.atan2(y, x)
  const bearing = (toDegrees(theta) + 360) % 360
  return bearing
}

function bearingToCompass(bearing) {
  const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
  const index = Math.round(bearing / 45) % directions.length
  return directions[index]
}

function toRadians(deg) {
  return (deg * Math.PI) / 180
}

function toDegrees(rad) {
  return (rad * 180) / Math.PI
}

window.addEventListener('beforeunload', () => {
  resizeObserver.disconnect()
})
